//! Escritura del texto en la ventana activa (== `pipeline/writer.py`).
//!
//! Igual que en Python, la lógica pura de reordenamiento (`SequenceBuffer`) está
//! separada de la I/O real de portapapeles/teclado, para poder testear el
//! algoritmo (la parte con más casos borde) sin tocar `enigo`/`arboard`.

use std::collections::HashMap;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::domain::models::CleanText;
use crate::domain::traits::{Clipboard, Keyboard};

/// Reordena `CleanText` que llegan fuera de orden y los libera en orden de `seq`.
#[derive(Default)]
pub struct SequenceBuffer {
    expected_seq: u64,
    pending: HashMap<u64, CleanText>,
}

impl SequenceBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Procesa un item y devuelve los que ya están listos, en orden contiguo.
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

/// Etapa del pipeline: reordena y pega cada texto simulando Ctrl+V.
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
                    // Texto vacío (audio que era pura muletilla): se saltea el
                    // pegado, pero el `seq` ya avanzó en el buffer.
                    tracing::info!(seq = ready.seq, "Skipping empty clean text");
                } else if let Err(error) = self.emit(&ready.text).await {
                    tracing::error!(%error, seq = ready.seq, "clipboard emit failed");
                }
            }
        }
    }

    async fn emit(&mut self, text: &str) -> anyhow::Result<()> {
        // Backup del portapapeles; se restaura SIEMPRE (== try/finally).
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

// --- Implementaciones reales (crean el handle por llamada para ser Send+Sync
// sin retener nada entre `.await`s). ---

/// Portapapeles del sistema vía `arboard`.
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

/// Teclado del sistema vía `enigo`. Simula Ctrl+V.
///
/// Nota heredada del proyecto Python: por UIPI, esto NO inyecta en ventanas
/// elevadas si el proceso no corre elevado (mismo requisito que con `pynput`).
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
        // Al llegar el 0 se drena 0,1,2 en orden.
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
        // Llega 0 -> drena 0,1 pero se detiene (falta el 2).
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

        // seq 0 vacío (se saltea), seq 1 con texto (se pega).
        tx.send(clean(0, "")).await.unwrap();
        tx.send(clean(1, "hola")).await.unwrap();
        drop(tx);

        writer.run(rx).await;

        // Solo un paste (el de seq 1), pese a que seq 0 llegó primero.
        assert_eq!(*pastes.lock().unwrap(), 1);
        // El texto pegado fue el de seq 1.
        assert!(cb.set_values.lock().unwrap().iter().any(|v| v == "hola"));
    }
}
