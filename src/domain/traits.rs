//! Injectable traits: seams for swapping every external dependency with a fake
//! in tests without touching production code.

use async_trait::async_trait;

/// Transcription backend, implemented by `GroqBackend`, `LocalWhisper` and test
/// fakes.
#[async_trait]
pub trait TranscriptionBackend: Send + Sync {
    /// Transcribes mono f32 PCM audio. The returned text may be empty.
    async fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: &str,
    ) -> anyhow::Result<String>;

    /// Releases the model if it has been idle for too long. Returns `true` when
    /// something was unloaded. Stateless backends (e.g. Groq) do nothing.
    async fn maybe_unload(&self) -> bool {
        false
    }

    /// Preloads the model so the first transcription does not pay the load cost
    /// (several seconds on GPU). Idempotent; stateless backends do nothing.
    async fn warm_up(&self) {}
}

/// Audio source. The real implementation wraps `cpal`; tests use a fake that
/// returns predefined samples.
pub trait AudioInput: Send {
    fn start(&self, sample_rate: u32, channels: u16) -> anyhow::Result<()>;

    /// Stops capture and returns the accumulated mono samples.
    fn stop(&self) -> anyhow::Result<Vec<f32>>;
}

/// System clipboard access. Real implementation: `arboard`.
pub trait Clipboard: Send {
    fn get_text(&mut self) -> anyhow::Result<String>;
    fn set_text(&mut self, text: &str) -> anyhow::Result<()>;
}

/// Keyboard simulation. Real implementation: `enigo`.
pub trait Keyboard: Send {
    /// Simulates the paste shortcut (Ctrl+V).
    fn paste(&mut self) -> anyhow::Result<()>;
}
