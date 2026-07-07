//! Traits inyectables. Son el equivalente de los `Protocol` de
//! `domain/protocols.py` combinados con las "fábricas inyectables" del proyecto
//! Python (`model_factory`, `stream_factory`, `pyperclip`/`pynput`): permiten
//! sustituir cada dependencia externa por un fake en los tests, sin tocar el
//! código de producción.

use async_trait::async_trait;

/// Backend de transcripción (== `model_factory` / la dependencia real de
/// `WhisperTranscriber`). Implementado por `GroqBackend` y `LocalWhisper`, y por
/// fakes en los tests.
#[async_trait]
pub trait TranscriptionBackend: Send + Sync {
    /// Transcribe audio PCM f32 mono. Devuelve el texto crudo (posiblemente vacío).
    async fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: &str,
    ) -> anyhow::Result<String>;

    /// Libera el modelo si lleva demasiado tiempo ocioso. Devuelve `true` si
    /// descargó algo. Los backends sin estado (p. ej. Groq) no hacen nada.
    async fn maybe_unload(&self) -> bool {
        false
    }

    /// Precarga el modelo para que la primera transcripción no pague el costo de
    /// carga (varios segundos en GPU). Idempotente y opcional: los backends sin
    /// estado (p. ej. Groq) no hacen nada.
    async fn warm_up(&self) {}
}

/// Fuente de audio (== `stream_factory` de `SoundDeviceRecorder`). La impl real
/// envuelve `cpal`; los tests usan un fake que devuelve muestras predefinidas.
pub trait AudioInput: Send {
    /// Inicia la captura a `sample_rate`/`channels`.
    fn start(&self, sample_rate: u32, channels: u16) -> anyhow::Result<()>;

    /// Detiene la captura y devuelve todas las muestras mono acumuladas.
    fn stop(&self) -> anyhow::Result<Vec<f32>>;
}

/// Acceso al portapapeles (== `pyperclip`). Impl real: `arboard`.
pub trait Clipboard: Send {
    fn get_text(&mut self) -> anyhow::Result<String>;
    fn set_text(&mut self, text: &str) -> anyhow::Result<()>;
}

/// Simulación de teclado (== `pynput`). Impl real: `enigo`.
pub trait Keyboard: Send {
    /// Simula el atajo de pegar (Ctrl+V).
    fn paste(&mut self) -> anyhow::Result<()>;
}
