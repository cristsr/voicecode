//! Transcription stage.
//!
//! Each chunk is transcribed on its own task, allowing overlap and out-of-order
//! completion (the writer's `SequenceBuffer` reorders). Concurrency is bounded
//! by a `Semaphore` (`max_workers`), and in-flight tasks are tracked in a
//! `JoinSet` so they can be drained on shutdown.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;

use crate::config::{Backend, Config};
use crate::domain::models::{AudioChunk, TranscribedText};
use crate::domain::traits::TranscriptionBackend;

#[cfg(feature = "groq")]
pub mod groq;

#[cfg(feature = "local")]
pub mod local;

/// How often the idle monitor checks whether the backend can be unloaded.
const IDLE_CHECK_INTERVAL: Duration = Duration::from_secs(30);

pub struct WhisperTranscriber {
    backend: Arc<dyn TranscriptionBackend>,
    language: String,
    semaphore: Arc<Semaphore>,
    idle_enabled: bool,
}

impl WhisperTranscriber {
    pub fn new(
        backend: Arc<dyn TranscriptionBackend>,
        language: String,
        max_workers: usize,
        idle_enabled: bool,
    ) -> Self {
        Self {
            backend,
            language,
            semaphore: Arc::new(Semaphore::new(max_workers.max(1))),
            idle_enabled,
        }
    }

    pub async fn run(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        text_tx: mpsc::Sender<TranscribedText>,
    ) {
        let mut tasks: JoinSet<()> = JoinSet::new();
        loop {
            tokio::select! {
                maybe_chunk = audio_rx.recv() => {
                    match maybe_chunk {
                        Some(chunk) => self.spawn_transcription(&mut tasks, chunk, &text_tx),
                        None => break, // channel closed: no more chunks
                    }
                }
                // Reap finished tasks so handles do not accumulate.
                Some(_) = tasks.join_next(), if !tasks.is_empty() => {}
            }
        }
        // Drain in-flight transcriptions before finishing.
        while tasks.join_next().await.is_some() {}
    }

    fn spawn_transcription(
        &self,
        tasks: &mut JoinSet<()>,
        chunk: AudioChunk,
        text_tx: &mpsc::Sender<TranscribedText>,
    ) {
        let backend = self.backend.clone();
        let semaphore = self.semaphore.clone();
        let language = self.language.clone();
        let text_tx = text_tx.clone();
        tasks.spawn(async move {
            let _permit = semaphore.acquire_owned().await.expect("semaphore closed");
            // Every chunk must produce a `TranscribedText`, even an empty one:
            // the writer's `SequenceBuffer` releases items strictly in `seq`
            // order, so a `seq` that never arrives here would permanently
            // stall every later chunk waiting behind it.
            let raw = match backend
                .transcribe(&chunk.data, chunk.sample_rate, &language)
                .await
            {
                Ok(raw) if !raw.trim().is_empty() => raw,
                Ok(_) => {
                    tracing::info!(seq = chunk.seq, "Discarding empty transcription");
                    String::new()
                }
                Err(error) => {
                    tracing::error!(%error, seq = chunk.seq, "Error transcribing audio chunk");
                    String::new()
                }
            };
            let _ = text_tx
                .send(TranscribedText {
                    seq: chunk.seq,
                    raw,
                })
                .await;
        });
    }

    /// Unloads the backend model once it has been idle for too long. Does
    /// nothing when idle-unload is disabled.
    pub async fn monitor_idle(&self) {
        if !self.idle_enabled {
            return;
        }
        loop {
            tokio::time::sleep(IDLE_CHECK_INTERVAL).await;
            self.backend.maybe_unload().await;
        }
    }

    /// Keeps the backend model warm: preloads it once at startup, then reloads
    /// it on every key-down ping (`warmup_rx`, fed by the `Recorder`) so a
    /// reload after an idle-unload overlaps with the user speaking instead of
    /// starting only once they release the key. `warm_up` is a cheap no-op once
    /// the model is already loaded, and a no-op altogether for stateless
    /// backends (e.g. Groq).
    pub async fn warm_up_loop(&self, mut warmup_rx: mpsc::Receiver<()>) {
        self.backend.warm_up().await;
        while warmup_rx.recv().await.is_some() {
            self.backend.warm_up().await;
        }
    }
}

/// Builds the backend selected in `config`, failing with a clear message when
/// it was not compiled in (feature disabled).
pub fn build_backend(config: &Config) -> anyhow::Result<Arc<dyn TranscriptionBackend>> {
    match config.transcriber.backend {
        Backend::Groq => {
            #[cfg(feature = "groq")]
            {
                Ok(Arc::new(groq::GroqBackend::from_config(config)?))
            }
            #[cfg(not(feature = "groq"))]
            {
                anyhow::bail!("backend 'groq' is not compiled in (rebuild with --features groq)")
            }
        }
        Backend::Local => {
            #[cfg(feature = "local")]
            {
                Ok(Arc::new(local::LocalWhisper::from_config(config)?))
            }
            #[cfg(not(feature = "local"))]
            {
                anyhow::bail!("backend 'local' is not compiled in (rebuild with --features local)")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    fn chunk(seq: u64) -> AudioChunk {
        AudioChunk {
            seq,
            data: vec![0.0; 1600],
            sample_rate: 16000,
        }
    }

    struct FakeBackend;
    #[async_trait]
    impl TranscriptionBackend for FakeBackend {
        async fn transcribe(&self, _: &[f32], _: u32, _: &str) -> anyhow::Result<String> {
            Ok("hola mundo".into())
        }
    }

    struct EmptyBackend;
    #[async_trait]
    impl TranscriptionBackend for EmptyBackend {
        async fn transcribe(&self, _: &[f32], _: u32, _: &str) -> anyhow::Result<String> {
            Ok("   ".into())
        }
    }

    struct ErrorBackend;
    #[async_trait]
    impl TranscriptionBackend for ErrorBackend {
        async fn transcribe(&self, _: &[f32], _: u32, _: &str) -> anyhow::Result<String> {
            anyhow::bail!("boom")
        }
    }

    async fn drain(
        backend: Arc<dyn TranscriptionBackend>,
        chunks: Vec<AudioChunk>,
    ) -> Vec<TranscribedText> {
        let transcriber = WhisperTranscriber::new(backend, "es".into(), 2, false);
        let (atx, arx) = mpsc::channel(16);
        let (ttx, mut trx) = mpsc::channel(16);
        for c in chunks {
            atx.send(c).await.unwrap();
        }
        drop(atx); // close the channel -> run() drains and returns
        transcriber.run(arx, ttx).await;
        let mut out = Vec::new();
        while let Ok(item) = trx.try_recv() {
            out.push(item);
        }
        out
    }

    #[tokio::test]
    async fn publishes_transcribed_text_for_valid_audio() {
        let out = drain(Arc::new(FakeBackend), vec![chunk(0)]).await;
        assert_eq!(
            out,
            vec![TranscribedText {
                seq: 0,
                raw: "hola mundo".into()
            }]
        );
    }

    #[tokio::test]
    async fn discards_empty_transcription() {
        // The seq must still arrive (with empty raw) so the writer's
        // SequenceBuffer does not stall waiting for it forever.
        let out = drain(Arc::new(EmptyBackend), vec![chunk(0)]).await;
        assert_eq!(
            out,
            vec![TranscribedText {
                seq: 0,
                raw: String::new()
            }]
        );
    }

    #[tokio::test]
    async fn survives_backend_error() {
        // Must neither hang nor panic, and must still emit the seq (empty)
        // so later chunks are not stuck behind it forever.
        let out = drain(Arc::new(ErrorBackend), vec![chunk(0)]).await;
        assert_eq!(
            out,
            vec![TranscribedText {
                seq: 0,
                raw: String::new()
            }]
        );
    }

    #[tokio::test]
    async fn overlapping_transcriptions_all_complete() {
        let out = drain(Arc::new(FakeBackend), vec![chunk(0), chunk(1), chunk(2)]).await;
        let seqs: std::collections::HashSet<u64> = out.iter().map(|t| t.seq).collect();
        assert_eq!(seqs, [0, 1, 2].into_iter().collect());
    }

    #[tokio::test]
    async fn warm_up_loop_preloads_then_warms_on_each_ping() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingBackend {
            warms: Arc<AtomicUsize>,
        }
        #[async_trait]
        impl TranscriptionBackend for CountingBackend {
            async fn transcribe(&self, _: &[f32], _: u32, _: &str) -> anyhow::Result<String> {
                Ok(String::new())
            }
            async fn warm_up(&self) {
                self.warms.fetch_add(1, Ordering::SeqCst);
            }
        }

        let warms = Arc::new(AtomicUsize::new(0));
        let transcriber = WhisperTranscriber::new(
            Arc::new(CountingBackend {
                warms: warms.clone(),
            }),
            "es".into(),
            2,
            false,
        );
        let (tx, rx) = mpsc::channel(1);
        tx.send(()).await.unwrap(); // one key-down ping
        drop(tx); // close the channel so the loop ends after draining

        transcriber.warm_up_loop(rx).await;

        // Preload once at startup + once per ping.
        assert_eq!(warms.load(Ordering::SeqCst), 2);
    }
}
