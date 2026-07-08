//! VoiceCode: push-to-talk dictation. A pipeline of decoupled stages connected
//! by `tokio::sync::mpsc` channels.

pub mod config;
pub mod domain;
pub mod pipeline;
pub mod utils;

use tokio::sync::mpsc;

use config::{Backend, Config};
use pipeline::cleaner::{compile_patterns, RegexCleaner};
use pipeline::listener::PttListener;
use pipeline::recorder::{CpalInput, Recorder};
use pipeline::transcriber::{self, WhisperTranscriber};
use pipeline::writer::{ClipboardWriter, SystemClipboard, SystemKeyboard};
use utils::platform::check_paste_dependencies;

/// Composition root: instantiates each stage with manually injected
/// dependencies, creates the pipeline's channels, and runs every stage
/// concurrently.
///
/// Does not return while the pipeline is alive (the `listener` stage never
/// completes).
pub async fn run_pipeline(config: Config) -> anyhow::Result<()> {
    for warning in check_paste_dependencies() {
        tracing::warn!("{warning}");
    }

    let backend = transcriber::build_backend(&config)?;
    let idle_enabled =
        config.transcriber.backend == Backend::Local && config.transcriber.idle_unload_seconds > 0;

    // Preload the model in the background so the first dictation does not wait
    // out the load time (a few seconds on GPU). Does not block pipeline
    // startup; stateless backends (Groq) ignore it.
    {
        let backend = backend.clone();
        tokio::spawn(async move { backend.warm_up().await });
    }

    let (key_tx, key_rx) = mpsc::channel(64);
    let (audio_tx, audio_rx) = mpsc::channel(16);
    let (text_tx, text_rx) = mpsc::channel(16);
    let (clean_tx, clean_rx) = mpsc::channel(16);

    let listener = PttListener::new(&config.ptt.key)?;
    let mut recorder = Recorder::new(
        config.audio.sample_rate,
        config.audio.channels,
        config.audio.min_audio_duration_ms,
        Box::new(CpalInput::new(config.audio.denoise)),
    );
    let transcriber = WhisperTranscriber::new(
        backend,
        config.whisper.language.clone(),
        config.transcriber.max_workers,
        idle_enabled,
    );
    let cleaner = RegexCleaner::new(compile_patterns(&config.cleaner.filler_patterns)?);
    let mut writer = ClipboardWriter::new(
        Box::new(SystemKeyboard),
        Box::new(SystemClipboard),
        config.writer.clipboard_restore_delay_ms,
    );

    tracing::info!(
        "VoiceCode iniciado - mantené {} para hablar",
        config.ptt.key
    );

    tokio::join!(
        listener.run(key_tx),
        recorder.run(key_rx, audio_tx),
        transcriber.run(audio_rx, text_tx),
        transcriber.monitor_idle(),
        cleaner.run(text_rx, clean_tx),
        writer.run(clean_rx),
    );

    Ok(())
}
