//! Etapa de transcripción (== `pipeline/transcriber.py`).
//!
//! Igual que en Python, cada chunk se transcribe en una task independiente
//! (permitiendo solapamiento y finalización fuera de orden; el `SequenceBuffer`
//! del writer reordena). La concurrencia se acota con un `Semaphore`
//! (== `transcriber_max_workers`), y las tasks en vuelo se rastrean en un
//! `JoinSet` (== el set `self._tasks`) para drenarlas al cerrar.

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

/// Cada cuánto revisa el monitor si el backend lleva demasiado tiempo ocioso.
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
                        None => break, // canal cerrado: no llegan más chunks
                    }
                }
                // Recolecta tasks completadas para no acumular handles.
                Some(_) = tasks.join_next(), if !tasks.is_empty() => {}
            }
        }
        // Drena las transcripciones en vuelo antes de terminar.
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
            match backend
                .transcribe(&chunk.data, chunk.sample_rate, &language)
                .await
            {
                Ok(raw) if !raw.trim().is_empty() => {
                    let _ = text_tx
                        .send(TranscribedText {
                            seq: chunk.seq,
                            raw,
                        })
                        .await;
                }
                Ok(_) => tracing::info!(seq = chunk.seq, "Discarding empty transcription"),
                Err(error) => {
                    tracing::error!(%error, seq = chunk.seq, "Error transcribing audio chunk")
                }
            }
        });
    }

    /// Descarga el modelo del backend cuando lleva demasiado tiempo ocioso.
    /// No hace nada si el idle-unload está deshabilitado.
    pub async fn monitor_idle(&self) {
        if !self.idle_enabled {
            return;
        }
        loop {
            tokio::time::sleep(IDLE_CHECK_INTERVAL).await;
            self.backend.maybe_unload().await;
        }
    }
}

/// Construye el backend indicado en `config`, fallando con un mensaje claro si
/// no fue compilado (feature apagada).
pub fn build_backend(config: &Config) -> anyhow::Result<Arc<dyn TranscriptionBackend>> {
    match config.transcriber.backend {
        Backend::Groq => {
            #[cfg(feature = "groq")]
            {
                Ok(Arc::new(groq::GroqBackend::from_config(config)?))
            }
            #[cfg(not(feature = "groq"))]
            {
                anyhow::bail!("backend 'groq' no está compilado (recompila con --features groq)")
            }
        }
        Backend::Local => {
            #[cfg(feature = "local")]
            {
                Ok(Arc::new(local::LocalWhisper::from_config(config)?))
            }
            #[cfg(not(feature = "local"))]
            {
                anyhow::bail!("backend 'local' no está compilado (recompila con --features local)")
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
        drop(atx); // cierra el canal -> run() drena y retorna
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
        let out = drain(Arc::new(EmptyBackend), vec![chunk(0)]).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn survives_backend_error() {
        // No debe colgarse ni paniquear; simplemente descarta el chunk.
        let out = drain(Arc::new(ErrorBackend), vec![chunk(0)]).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn overlapping_transcriptions_all_complete() {
        let out = drain(Arc::new(FakeBackend), vec![chunk(0), chunk(1), chunk(2)]).await;
        let seqs: std::collections::HashSet<u64> = out.iter().map(|t| t.seq).collect();
        assert_eq!(seqs, [0, 1, 2].into_iter().collect());
    }
}
