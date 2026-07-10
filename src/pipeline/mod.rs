//! The dictation pipeline: a chain of stages connected by `mpsc` channels.
//!
//! [`Pipeline`] is the composition root — [`Pipeline::build`] wires every stage
//! with its injected dependencies, and [`Pipeline::run`] creates the channels
//! and drives all stages concurrently. The process-level lifecycle (thread,
//! Tokio runtime, cancellation, restart) lives one level up in
//! [`crate::worker::Worker`].

pub mod cleaner;
pub mod listener;
pub mod recorder;
pub mod transcriber;
pub mod writer;

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::config::{Backend, Config};
use crate::utils::platform::check_paste_dependencies;
use cleaner::{compile_patterns, RegexCleaner};
use listener::PttListener;
use recorder::{CpalInput, Recorder};
use transcriber::WhisperTranscriber;
use writer::{ClipboardWriter, SystemClipboard, SystemKeyboard};

/// The assembled pipeline: every stage constructed with its dependencies,
/// ready to be run. Built by [`Pipeline::build`], driven by [`Pipeline::run`].
pub struct Pipeline {
    listener: PttListener,
    recorder: Recorder,
    transcriber: WhisperTranscriber,
    cleaner: RegexCleaner,
    writer: ClipboardWriter,
    ptt_key: String,
}

impl Pipeline {
    /// Wires every stage with manually injected dependencies. Fails if a stage's
    /// construction fails (backend selection, PTT key parsing, regex compile).
    pub fn build(config: &Config) -> anyhow::Result<Self> {
        for warning in check_paste_dependencies() {
            tracing::warn!("{warning}");
        }

        let backend = transcriber::build_backend(config)?;
        let idle_enabled = config.transcriber.backend == Backend::Local
            && config.transcriber.idle_unload_seconds > 0;

        let listener = PttListener::new(&config.ptt.key)?;
        let recorder = Recorder::new(
            config.audio.sample_rate,
            config.audio.channels,
            config.audio.min_audio_duration_ms,
            Arc::new(CpalInput::new(config.audio.denoise, config.audio.auto_gain)),
        );
        let transcriber = WhisperTranscriber::new(
            backend,
            config.whisper.language.clone(),
            config.transcriber.max_workers,
            idle_enabled,
        );
        let cleaner = RegexCleaner::new(compile_patterns(&config.cleaner.filler_patterns)?);
        let writer = ClipboardWriter::new(
            Box::new(SystemKeyboard),
            Box::new(SystemClipboard),
            config.writer.clipboard_restore_delay_ms,
        );

        Ok(Self {
            listener,
            recorder,
            transcriber,
            cleaner,
            writer,
            ptt_key: config.ptt.key.clone(),
        })
    }

    /// Creates the inter-stage channels and runs every stage concurrently.
    /// Does not return while the pipeline is alive (the `listener` stage never
    /// completes on its own).
    pub async fn run(self) {
        let Pipeline {
            listener,
            mut recorder,
            transcriber,
            cleaner,
            mut writer,
            ptt_key,
        } = self;

        tracing::info!("VoiceCode iniciado - mantené {ptt_key} para hablar");

        let (key_tx, key_rx) = mpsc::channel(64);
        let (audio_tx, audio_rx) = mpsc::channel(16);
        let (text_tx, text_rx) = mpsc::channel(16);
        let (clean_tx, clean_rx) = mpsc::channel(16);
        // Recorder pings this on every key-down; the transcriber's warm-up loop
        // consumes it so a model reload overlaps with the user speaking.
        let (warmup_tx, warmup_rx) = mpsc::channel::<()>(1);

        tokio::join!(
            listener.run(key_tx),
            recorder.run(key_rx, audio_tx, warmup_tx),
            transcriber.run(audio_rx, text_tx),
            transcriber.monitor_idle(),
            transcriber.warm_up_loop(warmup_rx),
            cleaner.run(text_rx, clean_tx),
            writer.run(clean_rx),
        );
    }
}
