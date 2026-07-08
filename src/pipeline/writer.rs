//! Writes the text into the active window.
//!
//! The pure reordering logic (`SequenceBuffer`) is separated from the real
//! clipboard/keyboard I/O so the algorithm (the part with the most edge cases)
//! can be tested without touching `enigo`/`arboard`.

use std::collections::HashMap;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::domain::models::CleanText;
use crate::domain::traits::{Clipboard, Keyboard};

/// Reorders `CleanText` arriving out of order and releases them by `seq`.
#[derive(Default)]
pub struct SequenceBuffer {
    expected_seq: u64,
    pending: HashMap<u64, CleanText>,
}

impl SequenceBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Processes an item and returns those now ready, in contiguous order.
    pub fn process(&mut self, item: CleanText) -> Vec<CleanText> {
        let mut ready = Vec::new();
        if item.seq == self.expected_seq {
            ready.push(item);
            self.expected_seq += 1;
            while let Some(next) = self.pending.remove(&self.expected_seq) {
                ready.push(next);
                self.expected_seq += 1;
            }
        } else {
            self.pending.insert(item.seq, item);
        }
        ready
    }
}

/// Pipeline stage: reorders and pastes each text by simulating Ctrl+V.
pub struct ClipboardWriter {
    keyboard: Box<dyn Keyboard>,
    clipboard: Box<dyn Clipboard>,
    buffer: SequenceBuffer,
    restore_delay: Duration,
}

impl ClipboardWriter {
    pub fn new(
        keyboard: Box<dyn Keyboard>,
        clipboard: Box<dyn Clipboard>,
        restore_delay_ms: u64,
    ) -> Self {
        Self {
            keyboard,
            clipboard,
            buffer: SequenceBuffer::new(),
            restore_delay: Duration::from_millis(restore_delay_ms),
        }
    }

    pub async fn run(&mut self, mut clean_rx: mpsc::Receiver<CleanText>) {
        while let Some(item) = clean_rx.recv().await {
            for ready in self.buffer.process(item) {
                if ready.text.is_empty() {
                    // Empty text (audio that was pure filler): skip the paste,
                    // but the `seq` has already advanced in the buffer.
                    tracing::info!(seq = ready.seq, "Skipping empty clean text");
                } else if let Err(error) = self.emit(&ready.text).await {
                    tracing::error!(%error, seq = ready.seq, "clipboard emit failed");
                }
            }
        }
    }

    async fn emit(&mut self, text: &str) -> anyhow::Result<()> {
        // Back up the clipboard; it is always restored, even on failure.
        let backup = self.clipboard.get_text().unwrap_or_default();

        let set_res = self.clipboard.set_text(text);
        let paste_res = if set_res.is_ok() {
            self.keyboard.paste()
        } else {
            Ok(())
        };

        tokio::time::sleep(self.restore_delay).await;
        let _ = self.clipboard.set_text(&backup);

        set_res?;
        paste_res?;
        Ok(())
    }
}

// --- Real implementations (a fresh handle per call keeps them Send+Sync
// without holding anything across `.await`s). ---

/// System clipboard via `arboard`.
pub struct SystemClipboard;

impl Clipboard for SystemClipboard {
    fn get_text(&mut self) -> anyhow::Result<String> {
        Ok(arboard::Clipboard::new()?.get_text().unwrap_or_default())
    }

    fn set_text(&mut self, text: &str) -> anyhow::Result<()> {
        arboard::Clipboard::new()?.set_text(text.to_string())?;
        Ok(())
    }
}

/// System keyboard via `enigo`. Simulates Ctrl+V.
///
/// Note: due to Windows UIPI this does NOT inject into elevated windows unless
/// the process itself runs elevated.
pub struct SystemKeyboard;

impl Keyboard for SystemKeyboard {
    fn paste(&mut self) -> anyhow::Result<()> {
        use enigo::{Direction, Enigo, Key, Keyboard as _, Settings};
        let mut enigo = Enigo::new(&Settings::default())?;
        enigo.key(Key::Control, Direction::Press)?;
        enigo.key(Key::Unicode('v'), Direction::Click)?;
        enigo.key(Key::Control, Direction::Release)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn clean(seq: u64, text: &str) -> CleanText {
        CleanText {
            seq,
            text: text.to_string(),
        }
    }

    // --- SequenceBuffer (puro) ---

    #[test]
    fn in_order_passes_through() {
        let mut buf = SequenceBuffer::new();
        assert_eq!(buf.process(clean(0, "a")), vec![clean(0, "a")]);
        assert_eq!(buf.process(clean(1, "b")), vec![clean(1, "b")]);
    }

    #[test]
    fn out_of_order_buffers_until_gap_fills() {
        let mut buf = SequenceBuffer::new();
        assert_eq!(buf.process(clean(1, "b")), vec![]);
        assert_eq!(buf.process(clean(2, "c")), vec![]);
        // Once 0 arrives, 0,1,2 drain in order.
        assert_eq!(
            buf.process(clean(0, "a")),
            vec![clean(0, "a"), clean(1, "b"), clean(2, "c")]
        );
    }

    #[test]
    fn stops_draining_at_first_gap() {
        let mut buf = SequenceBuffer::new();
        buf.process(clean(1, "b"));
        buf.process(clean(3, "d"));
        // 0 arrives -> drains 0,1 but stops (2 is missing).
        assert_eq!(
            buf.process(clean(0, "a")),
            vec![clean(0, "a"), clean(1, "b")]
        );
    }

    // --- ClipboardWriter (con fakes) ---

    #[derive(Clone, Default)]
    struct FakeKeyboard {
        pastes: Arc<Mutex<usize>>,
    }
    impl Keyboard for FakeKeyboard {
        fn paste(&mut self) -> anyhow::Result<()> {
            *self.pastes.lock().unwrap() += 1;
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct FakeClipboard {
        set_values: Arc<Mutex<Vec<String>>>,
    }
    impl Clipboard for FakeClipboard {
        fn get_text(&mut self) -> anyhow::Result<String> {
            Ok(String::new())
        }
        fn set_text(&mut self, text: &str) -> anyhow::Result<()> {
            self.set_values.lock().unwrap().push(text.to_string());
            Ok(())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn empty_text_skips_paste_but_seq_advances() {
        let kb = FakeKeyboard::default();
        let cb = FakeClipboard::default();
        let pastes = kb.pastes.clone();

        let mut writer = ClipboardWriter::new(Box::new(kb), Box::new(cb.clone()), 50);
        let (tx, rx) = mpsc::channel(8);

        // seq 0 empty (skipped), seq 1 with text (pasted).
        tx.send(clean(0, "")).await.unwrap();
        tx.send(clean(1, "hola")).await.unwrap();
        drop(tx);

        writer.run(rx).await;

        // Only one paste (seq 1), even though seq 0 arrived first.
        assert_eq!(*pastes.lock().unwrap(), 1);
        assert!(cb.set_values.lock().unwrap().iter().any(|v| v == "hola"));
    }
}
