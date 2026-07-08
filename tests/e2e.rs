//! Test de integración end-to-end (== `tests/test_pipeline_e2e.py`).
//!
//! Alimenta `AudioChunk` sintéticos y verifica que el texto se pega en el orden
//! correcto de `seq`, aun cuando las transcripciones terminan fuera de orden
//! (lo reordena el `SequenceBuffer` del writer). Usa fakes inyectados por los
//! traits, sin GPU, audio ni red.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;

use voicecode::domain::models::AudioChunk;
use voicecode::domain::traits::{Clipboard, Keyboard, TranscriptionBackend};
use voicecode::pipeline::cleaner::RegexCleaner;
use voicecode::pipeline::transcriber::WhisperTranscriber;
use voicecode::pipeline::writer::ClipboardWriter;

/// Backend fake: deriva el texto del `seq` (codificado en la primera muestra) y
/// duerme más para los `seq` bajos, forzando finalización en orden inverso.
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
    let cleaner = RegexCleaner::new(vec![]); // sin muletillas: texto intacto
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

    // Se envían en orden; el backend los completa en orden inverso.
    for seq in 0..3 {
        audio_tx.send(chunk(seq)).await.unwrap();
    }
    drop(audio_tx);

    tokio::join!(
        transcriber.run(audio_rx, text_tx),
        cleaner.run(text_rx, clean_tx),
        writer.run(clean_rx),
    );

    // Pese a terminar 2,1,0, el texto se pega en orden 0,1,2.
    assert_eq!(*pasted.lock().unwrap(), vec!["T0", "T1", "T2"]);
}
