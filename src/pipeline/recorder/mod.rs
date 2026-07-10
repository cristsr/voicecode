//! Audio recording stage.
//!
//! The `Recorder` is hardware-agnostic: it turns PTT key events into start/stop
//! commands on an injectable [`AudioInput`], tags each finished recording with a
//! sequence number, and forwards it downstream. The real hardware driver lives
//! in [`cpal_input`]; tests swap in a fake `AudioInput`.

mod cpal_input;

pub use cpal_input::CpalInput;

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::domain::models::{AudioChunk, KeyEvent, KeyEventKind};
use crate::domain::traits::AudioInput;
use crate::utils::audio::duration_ms;

pub struct Recorder {
    sample_rate: u32,
    channels: u16,
    min_audio_duration_ms: u32,
    input: Arc<dyn AudioInput>,
    seq: u64,
    recording: bool,
}

impl Recorder {
    pub fn new(
        sample_rate: u32,
        channels: u16,
        min_audio_duration_ms: u32,
        input: Arc<dyn AudioInput>,
    ) -> Self {
        Self {
            sample_rate,
            channels,
            min_audio_duration_ms,
            input,
            seq: 0,
            recording: false,
        }
    }

    /// `warmup_tx` is pinged on every genuine key-down (not key-repeat) so the
    /// transcription backend can start reloading its model, if idle-unloaded, in
    /// parallel with the user speaking instead of after they release the key.
    pub async fn run(
        &mut self,
        mut key_rx: mpsc::Receiver<KeyEvent>,
        audio_tx: mpsc::Sender<AudioChunk>,
        warmup_tx: mpsc::Sender<()>,
    ) {
        while let Some(event) = key_rx.recv().await {
            match event.kind {
                KeyEventKind::Down => self.start_recording(&warmup_tx),
                KeyEventKind::Up => self.stop_recording(&audio_tx).await,
            }
        }
    }

    fn start_recording(&mut self, warmup_tx: &mpsc::Sender<()>) {
        if self.recording {
            // OS key-repeat: ignore an extra `down` while already recording.
            tracing::debug!("Ignoring duplicate key-down while already recording (key repeat)");
            return;
        }
        // Fire-and-forget: a full channel or no receiver just means a warm-up
        // is already in flight or the backend does not need one (e.g. Groq).
        let _ = warmup_tx.try_send(());
        match self.input.start(self.sample_rate, self.channels) {
            Ok(()) => self.recording = true,
            Err(error) => tracing::error!(%error, "failed to start audio capture"),
        }
    }

    async fn stop_recording(&mut self, audio_tx: &mpsc::Sender<AudioChunk>) {
        if !self.recording {
            tracing::debug!("Ignoring key-up while not recording");
            return;
        }
        self.recording = false;

        // `stop()` blocks synchronously on a reply from the audio thread
        // (denoise + resample run there); run it on a blocking-pool thread so
        // it never ties up a Tokio worker thread while it waits.
        let input = self.input.clone();
        let result = tokio::task::spawn_blocking(move || input.stop()).await;
        let samples = match result {
            Ok(Ok(samples)) => samples,
            Ok(Err(error)) => {
                tracing::error!(%error, "failed to stop audio capture");
                return;
            }
            Err(error) => {
                tracing::error!(%error, "audio stop task panicked");
                return;
            }
        };

        if duration_ms(&samples, self.sample_rate) < self.min_audio_duration_ms as f64 {
            tracing::info!("Discarding recording shorter than min_audio_duration_ms");
            return;
        }

        let seq = self.seq;
        self.seq += 1;
        let _ = audio_tx
            .send(AudioChunk {
                seq,
                data: samples,
                sample_rate: self.sample_rate,
            })
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeInput {
        samples: Vec<f32>,
        starts: Arc<AtomicUsize>,
    }

    impl AudioInput for FakeInput {
        fn start(&self, _: u32, _: u16) -> anyhow::Result<()> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn stop(&self) -> anyhow::Result<Vec<f32>> {
            Ok(self.samples.clone())
        }
    }

    fn recorder(samples: Vec<f32>, starts: Arc<AtomicUsize>) -> Recorder {
        Recorder::new(16000, 1, 300, Arc::new(FakeInput { samples, starts }))
    }

    fn down() -> KeyEvent {
        KeyEvent {
            kind: KeyEventKind::Down,
            key: "f12".into(),
        }
    }
    fn up() -> KeyEvent {
        KeyEvent {
            kind: KeyEventKind::Up,
            key: "f12".into(),
        }
    }

    async fn run_events(mut rec: Recorder, events: Vec<KeyEvent>) -> Vec<AudioChunk> {
        let (ktx, krx) = mpsc::channel(16);
        let (atx, mut arx) = mpsc::channel(16);
        // Keep the warm-up receiver alive for the duration of the run so the
        // recorder's fire-and-forget pings have somewhere to land.
        let (warmup_tx, _warmup_rx) = mpsc::channel(1);
        for e in events {
            ktx.send(e).await.unwrap();
        }
        drop(ktx);
        rec.run(krx, atx, warmup_tx).await;
        let mut out = Vec::new();
        while let Ok(chunk) = arx.try_recv() {
            out.push(chunk);
        }
        out
    }

    #[tokio::test]
    async fn seq_increments_per_recording() {
        let starts = Arc::new(AtomicUsize::new(0));
        let rec = recorder(vec![0.0; 16000], starts.clone());
        let out = run_events(rec, vec![down(), up(), down(), up()]).await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].seq, 0);
        assert_eq!(out[1].seq, 1);
    }

    #[tokio::test]
    async fn discards_recording_shorter_than_minimum() {
        let starts = Arc::new(AtomicUsize::new(0));
        // 160 samples at 16 kHz = 10 ms < 300 ms.
        let rec = recorder(vec![0.0; 160], starts.clone());
        let out = run_events(rec, vec![down(), up()]).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn ignores_duplicate_key_down() {
        let starts = Arc::new(AtomicUsize::new(0));
        let rec = recorder(vec![0.0; 16000], starts.clone());
        let out = run_events(rec, vec![down(), down(), up()]).await;
        assert_eq!(out.len(), 1);
        // Capture started only once despite the two `down` events.
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn ignores_key_up_without_down() {
        let starts = Arc::new(AtomicUsize::new(0));
        let rec = recorder(vec![0.0; 16000], starts.clone());
        let out = run_events(rec, vec![up()]).await;
        assert!(out.is_empty());
        assert_eq!(starts.load(Ordering::SeqCst), 0);
    }
}
