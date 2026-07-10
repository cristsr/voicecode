//! Process-level lifecycle for the pipeline.
//!
//! [`Worker`] owns the Tokio runtime and the thread it runs on, and exposes
//! `start` / `stop` / `restart` so the UI layer (the tray menu) can control the
//! pipeline without touching Tokio, cancellation, or the pipeline's internals.

use std::thread::JoinHandle;

use tokio_util::sync::CancellationToken;

use crate::config::Config;
use crate::pipeline::Pipeline;

/// Runs the pipeline on its own Tokio runtime, on a dedicated thread, and
/// exposes a handle to cancel or restart it.
pub struct Worker {
    cancel: CancellationToken,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Spawns the worker thread, builds its own Tokio runtime, loads the config
    /// and runs the pipeline until cancelled.
    pub fn start() -> Self {
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let handle = std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(error) => {
                    tracing::error!(%error, "failed to build Tokio runtime");
                    return;
                }
            };
            runtime.block_on(async move {
                let config = match Config::load_default() {
                    Ok(config) => config,
                    Err(error) => {
                        tracing::error!(%error, "failed to load config");
                        return;
                    }
                };
                let pipeline = match Pipeline::build(&config) {
                    Ok(pipeline) => pipeline,
                    Err(error) => {
                        tracing::error!(%error, "failed to build pipeline");
                        return;
                    }
                };
                tokio::select! {
                    _ = pipeline.run() => {}
                    _ = token.cancelled() => tracing::info!("pipeline cancelled"),
                }
            });
        });
        Self {
            cancel,
            handle: Some(handle),
        }
    }

    /// Cancels the pipeline and waits for the worker thread to finish.
    pub fn stop(&mut self) {
        self.cancel.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Stops the current pipeline and starts a fresh one (e.g. to pick up config
    /// changes from the tray's *Restart* action).
    pub fn restart(&mut self) {
        self.stop();
        *self = Self::start();
    }
}
