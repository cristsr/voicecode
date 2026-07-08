//! Local transcription backend using whisper.cpp (via `whisper-rs`).
//!
//! The model is loaded lazily and unloaded from the GPU after an idle period.
//! Inference is blocking, so it runs on `spawn_blocking`.
//!
//! Requires building with `--features local` (needs CMake and a CUDA toolchain).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::config::Config;
use crate::domain::traits::TranscriptionBackend;

/// A loaded model together with a pool of inference states for it.
///
/// Creating a `WhisperState` allocates ~700MB of GPU buffers (KV cache +
/// compute buffers), so states are checked out of the pool and returned after
/// use instead of being recreated on every transcription. The pool lives and
/// dies with this `Arc`: when the model is idle-unloaded (`Inner::loaded` set
/// to `None`) and no in-flight call still holds a clone of it, dropping the
/// last reference frees both the pooled states and the context's VRAM.
struct LoadedModel {
    ctx: WhisperContext,
    states: Mutex<Vec<WhisperState>>,
}

impl LoadedModel {
    fn checkout_state(&self) -> anyhow::Result<WhisperState> {
        if let Some(state) = self.states.lock().unwrap().pop() {
            return Ok(state);
        }
        Ok(self.ctx.create_state()?)
    }

    fn checkin_state(&self, state: WhisperState) {
        self.states.lock().unwrap().push(state);
    }
}

struct Inner {
    /// `None` while the model is unloaded (lazy loading).
    loaded: Option<Arc<LoadedModel>>,
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
        // Redirects whisper.cpp/GGML's native diagnostic logs (backend
        // selection, VRAM allocation, CUDA init failures) into `tracing`,
        // where they land in the same log the rest of the app uses. Without
        // this they print to stderr, which does not exist in the windowless
        // release build. Idempotent: safe if called more than once.
        whisper_rs::install_logging_hooks();

        let model_path = config.whisper.model_path.clone();
        if model_path.is_empty() {
            anyhow::bail!("local backend: set [whisper] model_path to the ggml (.bin) model path");
        }
        Ok(Self {
            model_path,
            idle_unload: Duration::from_secs(config.transcriber.idle_unload_seconds),
            inner: Mutex::new(Inner {
                loaded: None,
                active: 0,
                last_used: Instant::now(),
            }),
            load_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// Returns the loaded model (loading it on demand) and marks a
    /// transcription in flight so the monitor does not unload it in the
    /// meantime.
    ///
    /// Loading (several seconds on GPU) runs on `spawn_blocking` and is
    /// serialized with `load_lock`. The `std::Mutex` on `inner` is never held
    /// during the load nor across an `.await`, so recordings arriving while it
    /// loads are queued and processed rather than lost.
    async fn acquire(&self) -> anyhow::Result<Arc<LoadedModel>> {
        // Fast path: already loaded.
        if let Some(loaded) = self.loaded_model() {
            return Ok(loaded);
        }
        // Slow path: load. Serialized to avoid a double load.
        let _guard = self.load_lock.lock().await;
        // Re-check: another task may have loaded it while we waited for the lock.
        if let Some(loaded) = self.loaded_model() {
            return Ok(loaded);
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

        let loaded = Arc::new(LoadedModel {
            ctx,
            states: Mutex::new(Vec::new()),
        });
        // Pre-create one state so the first real transcription does not also
        // pay the ~700MB GPU buffer allocation on top of the model load.
        match loaded.checkout_state() {
            Ok(state) => loaded.checkin_state(state),
            Err(error) => tracing::warn!(%error, "failed to pre-create initial Whisper state"),
        }
        tracing::info!("Whisper model preloaded");

        let mut inner = self.inner.lock().unwrap();
        inner.loaded = Some(loaded.clone());
        inner.active += 1;
        Ok(loaded)
    }

    /// When the model is loaded, counts an in-flight transcription and returns
    /// it; otherwise `None`.
    fn loaded_model(&self) -> Option<Arc<LoadedModel>> {
        let mut inner = self.inner.lock().unwrap();
        let loaded = inner.loaded.clone()?;
        inner.active += 1;
        Some(loaded)
    }

    fn release(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.active = inner.active.saturating_sub(1);
        inner.last_used = Instant::now();
    }
}

/// Threads whisper.cpp uses for CPU inference. Leaves a couple of cores free
/// for the rest of the pipeline (and the OS) instead of claiming every logical
/// core, which is what pins the CPU at 100% and makes the whole machine feel
/// sluggish while a chunk is transcribing. Only matters on the CPU fallback
/// path; GPU inference barely touches the CPU.
fn inference_thread_count() -> i32 {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    available.saturating_sub(2).max(1) as i32
}

/// Blocking inference (runs on a `spawn_blocking` thread). `state` is checked
/// out of (and, by the caller, back into) the model's state pool rather than
/// created fresh each call.
fn run_inference(state: &mut WhisperState, audio: &[f32], language: &str) -> anyhow::Result<String> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(language));
    params.set_n_threads(inference_thread_count());
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
        let loaded = self.acquire().await?;
        let state = loaded.checkout_state()?;
        let audio = audio.to_vec();
        let language = language.to_string();

        let joined = tokio::task::spawn_blocking(move || {
            let mut state = state;
            let result = run_inference(&mut state, &audio, &language);
            (state, result)
        })
        .await;

        // Always release, even if inference failed.
        self.release();

        match joined {
            Ok((state, result)) => {
                // Return the state to the pool so the next transcription
                // reuses its GPU buffers instead of reallocating them.
                loaded.checkin_state(state);
                result
            }
            Err(error) => Err(anyhow::anyhow!("inference task panicked: {error}")),
        }
    }

    async fn warm_up(&self) {
        // Called both at startup and on every PTT key-down (see
        // `Recorder`), so the model reload after an idle-unload overlaps
        // with the user speaking instead of happening after they release the
        // key. `acquire()` logs and does real work only the first time (the
        // actual load); once loaded, this is just a cheap refcount bump.
        match self.acquire().await {
            Ok(_) => self.release(),
            Err(error) => tracing::error!(%error, "failed to preload Whisper model"),
        }
    }

    async fn maybe_unload(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let idle = inner.last_used.elapsed();
        if inner.loaded.is_some() && inner.active == 0 && idle >= self.idle_unload {
            tracing::info!("Unloading idle Whisper model after {:?}", idle);
            // Dropping the last reference frees the pooled states' and the
            // context's VRAM.
            inner.loaded = None;
            true
        } else {
            false
        }
    }
}
