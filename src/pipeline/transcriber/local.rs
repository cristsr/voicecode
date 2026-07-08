//! Backend de transcripción local con whisper.cpp (vía `whisper-rs`).
//!
//! Replica la optimización del proyecto Python: carga perezosa del modelo y
//! descarga de la GPU tras un periodo de inactividad (== `_acquire_model` /
//! `_release_model` / `_maybe_unload` / `monitor_idle`). La inferencia es
//! bloqueante, así que corre en `spawn_blocking` (== `run_in_executor`).
//!
//! Requiere compilar con `--features local` (necesita CMake y toolchain CUDA).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::Config;
use crate::domain::traits::TranscriptionBackend;

struct Inner {
    /// `None` mientras el modelo está descargado (carga perezosa).
    ctx: Option<Arc<WhisperContext>>,
    /// Transcripciones en curso; el modelo no se descarga si es > 0.
    active: usize,
    last_used: Instant,
}

pub struct LocalWhisper {
    model_path: String,
    idle_unload: Duration,
    inner: Mutex<Inner>,
    /// Serializa la carga del modelo para no cargarlo dos veces si llegan varias
    /// transcripciones a la vez con el modelo descargado.
    load_lock: tokio::sync::Mutex<()>,
}

impl LocalWhisper {
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let model_path = config.whisper.model_path.clone();
        if model_path.is_empty() {
            anyhow::bail!(
                "backend local: definí [whisper] model_path con la ruta al modelo ggml (.bin)"
            );
        }
        Ok(Self {
            model_path,
            idle_unload: Duration::from_secs(config.transcriber.idle_unload_seconds),
            inner: Mutex::new(Inner {
                ctx: None,
                active: 0,
                last_used: Instant::now(),
            }),
            load_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// Devuelve el contexto (cargándolo bajo demanda) y marca una transcripción
    /// en curso para que el monitor no lo descargue mientras tanto.
    ///
    /// La carga (varios segundos en GPU) corre en `spawn_blocking` para no bloquear
    /// el runtime, y se serializa con `load_lock`; el `std::Mutex` de `inner` nunca
    /// se sostiene durante la carga ni cruza un `.await`, así las grabaciones que
    /// lleguen mientras carga se encolan y se procesan (no se pierden).
    async fn acquire(&self) -> anyhow::Result<Arc<WhisperContext>> {
        // Camino rápido: ya está cargado.
        if let Some(ctx) = self.loaded_ctx() {
            return Ok(ctx);
        }
        // Camino lento: cargar. Serializado para evitar doble carga.
        let _guard = self.load_lock.lock().await;
        // Re-chequear: otra tarea pudo cargarlo mientras esperábamos el lock.
        if let Some(ctx) = self.loaded_ctx() {
            return Ok(ctx);
        }
        let path = self.model_path.clone();
        tracing::info!(path = %path, "Loading Whisper model (GPU-first)");
        // GPU-first: usa la GPU si whisper.cpp se compiló con soporte (feature
        // `cuda`) y hay dispositivo; si no, cae a CPU. Sin feature GPU es no-op.
        let ctx = tokio::task::spawn_blocking(move || {
            let mut params = WhisperContextParameters::default();
            params.use_gpu(true);
            WhisperContext::new_with_params(&path, params)
        })
        .await
        .map_err(|error| anyhow::anyhow!("model load task panicked: {error}"))??;

        let ctx = Arc::new(ctx);
        let mut inner = self.inner.lock().unwrap();
        inner.ctx = Some(ctx.clone());
        inner.active += 1;
        Ok(ctx)
    }

    /// Si el modelo está cargado, cuenta una transcripción en curso y devuelve el
    /// contexto; si no, `None`.
    fn loaded_ctx(&self) -> Option<Arc<WhisperContext>> {
        let mut inner = self.inner.lock().unwrap();
        let ctx = inner.ctx.clone()?;
        inner.active += 1;
        Some(ctx)
    }

    fn release(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.active = inner.active.saturating_sub(1);
        inner.last_used = Instant::now();
    }
}

/// Inferencia bloqueante (se ejecuta en un thread de `spawn_blocking`).
fn run_inference(ctx: &WhisperContext, audio: &[f32], language: &str) -> anyhow::Result<String> {
    let mut state = ctx.create_state()?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(language));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    state.full(params, audio)?;

    let segments = state.full_n_segments()?;
    let mut text = String::new();
    for i in 0..segments {
        text.push_str(&state.full_get_segment_text(i)?);
    }
    Ok(text.trim().to_string())
}

#[async_trait]
impl TranscriptionBackend for LocalWhisper {
    async fn transcribe(
        &self,
        audio: &[f32],
        _sample_rate: u32,
        language: &str,
    ) -> anyhow::Result<String> {
        let ctx = self.acquire().await?;
        let audio = audio.to_vec();
        let language = language.to_string();

        let joined =
            tokio::task::spawn_blocking(move || run_inference(&ctx, &audio, &language)).await;

        // Liberar SIEMPRE, aunque la inferencia falle (== try/finally).
        self.release();
        joined?
    }

    async fn warm_up(&self) {
        match self.acquire().await {
            Ok(_) => {
                self.release();
                tracing::info!("Whisper model preloaded");
            }
            Err(error) => tracing::error!(%error, "failed to preload Whisper model"),
        }
    }

    async fn maybe_unload(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let idle = inner.last_used.elapsed();
        if inner.ctx.is_some() && inner.active == 0 && idle >= self.idle_unload {
            tracing::info!("Unloading idle Whisper model after {:?}", idle);
            // Soltar la referencia libera la VRAM al destruirse el contexto.
            inner.ctx = None;
            true
        } else {
            false
        }
    }
}
