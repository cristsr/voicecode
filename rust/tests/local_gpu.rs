//! Prueba de integración del backend local (whisper.cpp) contra el modelo GGML
//! real. Verifica de punta a punta que el modelo carga y transcribe; con la
//! feature `local` (CUDA) además debería inicializar la GPU (mirar el stderr de
//! whisper.cpp con `cargo test --features local -- --nocapture`).
//!
//! Se **salta automáticamente** si no está el modelo en `models/ggml-large-v3.bin`
//! (o en la ruta de la env var `VOICECODE_TEST_MODEL`), así el resto de CI no
//! depende de un archivo de ~3 GB.

#![cfg(any(feature = "local", feature = "local-cpu"))]

use std::path::PathBuf;

use voicecode::config::Config;
use voicecode::domain::traits::TranscriptionBackend;
use voicecode::pipeline::transcriber::local::LocalWhisper;

/// Ruta al modelo GGML: env var o el default dentro del repo.
fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("VOICECODE_TEST_MODEL") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/ggml-large-v3.bin")
}

/// Lee un WAV PCM de 16 bits mono a `Vec<f32>` normalizado a [-1, 1].
/// Parser mínimo: busca el chunk `data` y lee muestras `i16` LE.
fn read_wav_i16_mono(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("no se pudo leer el WAV de muestra");
    // Encuentra el chunk "data": 4 bytes tag + 4 bytes tamaño, luego las muestras.
    let pos = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("WAV sin chunk data");
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
            "SKIP: falta el modelo ({}) o el sample ({}). \
             Descargá el GGML para correr esta prueba.",
            model.display(),
            sample.display()
        );
        return;
    }

    let mut config = Config::default();
    config.whisper.model_path = model.to_string_lossy().into_owned();
    config.transcriber.idle_unload_seconds = 0; // no descargar durante el test

    let backend = LocalWhisper::from_config(&config).expect("construir LocalWhisper");

    let audio = read_wav_i16_mono(&sample);
    assert!(!audio.is_empty(), "el sample no tiene muestras");

    let text = backend
        .transcribe(&audio, 16_000, "en")
        .await
        .expect("la transcripción falló");

    eprintln!("== Transcripción del modelo: {text:?}");
    let lower = text.to_lowercase();
    // jfk.wav: "...ask not what your country can do for you..."
    assert!(
        lower.contains("country") || lower.contains("ask"),
        "transcripción inesperada: {text:?}"
    );
}
