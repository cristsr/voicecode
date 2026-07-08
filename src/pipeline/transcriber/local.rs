//! Local transcription backend using whisper.cpp (via `whisper-rs`).
//!
//! The model is loaded lazily and unloaded from the GPU after an idle period.
//! Inference is blocking, so it runs on `spawn_blocking`.
//!
//! Requires building with `--features local` (needs CMake and a CUDA toolchain).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::Config;
use crate::domain::traits::TranscriptionBackend;

struct Inner {
    /// `None` while the model is unloaded (lazy loading).
    ctx: Option<Arc<WhisperContext>>,
    /// In-flight transcriptions; the model is not unloaded while this is > 0.
    active: usize,
    last_used: Instant,
}

pub struct LocalWhisper {
    model_path: String,
    idle_unload: Duration,
    inner: Mutex<Inner>,
    /// Serializes model loading so it is not loaded twice when several
    /// transcriptions arrive at once with the model unloaded.
    load_lock: tokio::sync::Mutex<()>,
}

impl LocalWhisper {
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let model_path = config.whisper.model_path.clone();
        if model_path.is_empty() {
            anyhow::bail!("local backend: set [whisper] model_path to the ggml (.bin) model path");
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

    /// Returns the context (loading it on demand) and marks a transcription in
    /// flight so the monitor does not unload it in the meantime.
    ///
    /// Loading (several seconds on GPU) runs on `spawn_blocking` and is
    /// serialized with `load_lock`. The `std::Mutex` on `inner` is never held
    /// during the load nor across an `.await`, so recordings arriving while it
    /// loads are queued and processed rather than lost.
    async fn acquire(&self) -> anyhow::Result<Arc<WhisperContext>> {
        // Fast path: already loaded.
        if let Some(ctx) = self.loaded_ctx() {
            return Ok(ctx);
        }
        // Slow path: load. Serialized to avoid a double load.
        let _guard = self.load_lock.lock().await;
        // Re-check: another task may have loaded it while we waited for the lock.
        if let Some(ctx) = self.loaded_ctx() {
            return Ok(ctx);
        }
        let path = self.model_path.clone();
        tracing::info!(path = %path, "Loading Whisper model (GPU-first)");
        // GPU-first: uses the GPU when whisper.cpp was built with the `cuda`
        // feature and a device exists; otherwise falls back to CPU.
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

    /// When the model is loaded, counts an in-flight transcription and returns
    /// the context; otherwise `None`.
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

/// Blocking inference (runs on a `spawn_blocking` thread).
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

        // Always release, even if inference failed.
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
            // Dropping the reference frees the VRAM when the context is destroyed.
            inner.ctx = None;
            true
        } else {
            false
        }
    }
}
