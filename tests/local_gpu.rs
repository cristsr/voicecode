//! Integration test for the local backend (whisper.cpp) against a real GGML
//! model. Verifies end-to-end that the model loads and transcribes; with the
//! `local` (CUDA) feature it should also initialize the GPU — check
//! whisper.cpp's stderr via `cargo test --features local -- --nocapture`.
//!
//! Automatically **skipped** when the model is missing from
//! `models/ggml-large-v3.bin` (or the path in `VOICECODE_TEST_MODEL`), so the
//! rest of CI does not depend on a ~3 GB file.

#![cfg(any(feature = "local", feature = "local-cpu"))]

use std::path::PathBuf;

use voicecode::config::Config;
use voicecode::domain::traits::TranscriptionBackend;
use voicecode::pipeline::transcriber::local::LocalWhisper;

/// Path to the GGML model: env var override or the repo default.
fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("VOICECODE_TEST_MODEL") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/ggml-large-v3.bin")
}

/// Reads a 16-bit mono PCM WAV into `Vec<f32>` normalized to `[-1, 1]`.
/// Minimal parser: finds the `data` chunk and reads little-endian `i16` samples.
fn read_wav_i16_mono(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("failed to read the sample WAV");
    // Find the "data" chunk: 4-byte tag + 4-byte size, then the samples.
    let pos = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("WAV has no data chunk");
    let data_start = pos + 8;
    bytes[data_start..]
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect()
}

#[tokio::test]
async fn transcribes_sample_audio_with_real_model() {
    let model = model_path();
    let sample = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/jfk.wav");

    if !model.exists() || !sample.exists() {
        eprintln!(
            "SKIP: missing model ({}) or sample ({}). \
             Download the GGML model to run this test.",
            model.display(),
            sample.display()
        );
        return;
    }

    let mut config = Config::default();
    config.whisper.model_path = model.to_string_lossy().into_owned();
    config.transcriber.idle_unload_seconds = 0; // do not unload during the test

    let backend = LocalWhisper::from_config(&config).expect("build LocalWhisper");

    let audio = read_wav_i16_mono(&sample);
    assert!(!audio.is_empty(), "sample has no audio data");

    let text = backend
        .transcribe(&audio, 16_000, "en")
        .await
        .expect("transcription failed");

    eprintln!("== Model transcription: {text:?}");
    let lower = text.to_lowercase();
    // jfk.wav: "...ask not what your country can do for you..."
    assert!(
        lower.contains("country") || lower.contains("ask"),
        "unexpected transcription: {text:?}"
    );
}
