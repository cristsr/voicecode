//! End-to-end integration test.
//!
//! Feeds synthetic `AudioChunk`s and verifies text is pasted in the correct
//! `seq` order even when transcriptions finish out of order (reordered by the
//! writer's `SequenceBuffer`). Uses fakes injected via the traits — no GPU,
//! audio or network involved.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;

use voicecode::domain::models::AudioChunk;
use voicecode::domain::traits::{Clipboard, Keyboard, TranscriptionBackend};
use voicecode::pipeline::cleaner::RegexCleaner;
use voicecode::pipeline::transcriber::WhisperTranscriber;
use voicecode::pipeline::writer::ClipboardWriter;

/// Fake backend: derives the text from `seq` (encoded in the first sample) and
/// sleeps longer for lower `seq` values, forcing reverse-order completion.
struct SeqBackend;

#[async_trait]
impl TranscriptionBackend for SeqBackend {
    async fn transcribe(&self, audio: &[f32], _: u32, _: &str) -> anyhow::Result<String> {
        let seq = audio[0] as u64;
        tokio::time::sleep(Duration::from_millis((3 - seq) * 20)).await;
        Ok(format!("t{seq}"))
    }
}

#[derive(Clone)]
struct FakeClipboard {
    current: Arc<Mutex<String>>,
}
impl Clipboard for FakeClipboard {
    fn get_text(&mut self) -> anyhow::Result<String> {
        Ok(self.current.lock().unwrap().clone())
    }
    fn set_text(&mut self, text: &str) -> anyhow::Result<()> {
        *self.current.lock().unwrap() = text.to_string();
        Ok(())
    }
}

#[derive(Clone)]
struct FakeKeyboard {
    current: Arc<Mutex<String>>,
    pasted: Arc<Mutex<Vec<String>>>,
}
impl Keyboard for FakeKeyboard {
    fn paste(&mut self) -> anyhow::Result<()> {
        let text = self.current.lock().unwrap().clone();
        self.pasted.lock().unwrap().push(text);
        Ok(())
    }
}

fn chunk(seq: u64) -> AudioChunk {
    AudioChunk {
        seq,
        data: vec![seq as f32; 1600],
        sample_rate: 16000,
    }
}

#[tokio::test]
async fn pipeline_emits_text_in_seq_order_despite_out_of_order_transcription() {
    let current = Arc::new(Mutex::new(String::new()));
    let pasted = Arc::new(Mutex::new(Vec::new()));

    let transcriber = WhisperTranscriber::new(Arc::new(SeqBackend), "es".into(), 8, false);
    let cleaner = RegexCleaner::new(vec![]); // no filler patterns: text passes through untouched
    let mut writer = ClipboardWriter::new(
        Box::new(FakeKeyboard {
            current: current.clone(),
            pasted: pasted.clone(),
        }),
        Box::new(FakeClipboard {
            current: current.clone(),
        }),
        0,
    );

    let (audio_tx, audio_rx) = mpsc::channel(16);
    let (text_tx, text_rx) = mpsc::channel(16);
    let (clean_tx, clean_rx) = mpsc::channel(16);

    // Sent in order; the backend completes them in reverse order.
    for seq in 0..3 {
        audio_tx.send(chunk(seq)).await.unwrap();
    }
    drop(audio_tx);

    tokio::join!(
        transcriber.run(audio_rx, text_tx),
        cleaner.run(text_rx, clean_tx),
        writer.run(clean_rx),
    );

    // Despite finishing 2,1,0, the text is pasted in order 0,1,2.
    assert_eq!(*pasted.lock().unwrap(), vec!["T0", "T1", "T2"]);
}
