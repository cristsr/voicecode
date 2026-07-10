//! Configuration loading. Reads a sectioned TOML and falls back to defaults for
//! any missing section or key (or when the file is absent). The `#[serde(default)]`
//! on every level lets a partial TOML merge with the defaults automatically.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Transcription backend, selected in `[transcriber] backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Local,
    Groq,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Ptt {
    pub key: String,
}

impl Default for Ptt {
    fn default() -> Self {
        Self { key: "f12".into() }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Audio {
    pub sample_rate: u32,
    pub channels: u16,
    pub min_audio_duration_ms: u32,
    /// Apply RNNoise suppression to the recording before transcribing.
    pub denoise: bool,
    /// Auto-normalize the recording's level toward a consistent speech loudness
    /// before transcribing, so a far-away mic does not force raising the voice.
    pub auto_gain: bool,
}

impl Default for Audio {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            min_audio_duration_ms: 300,
            denoise: true,
            auto_gain: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Whisper {
    pub model: String,
    pub device: String,
    pub compute_type: String,
    pub language: String,
    /// Path to the GGML (.bin) model for the local backend. Only used when
    /// `backend = "local"`.
    pub model_path: String,
}

impl Default for Whisper {
    fn default() -> Self {
        Self {
            model: "large-v3".into(),
            device: "cuda".into(),
            compute_type: "float16".into(),
            language: "es".into(),
            model_path: String::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Transcriber {
    pub max_workers: usize,
    /// Unload the model from the GPU after N idle seconds (0 = never).
    pub idle_unload_seconds: u64,
    pub backend: Backend,
}

impl Default for Transcriber {
    fn default() -> Self {
        Self {
            max_workers: 2,
            idle_unload_seconds: 300,
            backend: Backend::Groq,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Cleaner {
    pub filler_patterns: Vec<String>,
}

impl Default for Cleaner {
    fn default() -> Self {
        Self {
            filler_patterns: vec![
                r"\beh+\b".into(),
                r"\bmmm+\b".into(),
                r"\bo sea\b".into(),
                r"\bdigamos\b".into(),
                r"\bbásicamente\b".into(),
                r"\bpues\b".into(),
                r"\benton?ces\b".into(),
                r"\bla verdad\b".into(),
            ],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Writer {
    pub clipboard_restore_delay_ms: u64,
}

impl Default for Writer {
    fn default() -> Self {
        Self {
            clipboard_restore_delay_ms: 50,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Groq {
    pub model: String,
    /// Name of the environment variable holding the API key. The key is never
    /// read from the TOML so secrets are not versioned.
    pub api_key_env: String,
    pub base_url: String,
}

impl Default for Groq {
    fn default() -> Self {
        Self {
            model: "whisper-large-v3".into(),
            api_key_env: "GROQ_API_KEY".into(),
            base_url: "https://api.groq.com/openai/v1".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub ptt: Ptt,
    pub audio: Audio,
    pub whisper: Whisper,
    pub transcriber: Transcriber,
    pub cleaner: Cleaner,
    pub writer: Writer,
    pub groq: Groq,
}

impl Config {
    /// Loads from a TOML file, returning the defaults when it does not exist.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)?;
        let config = toml::from_str(&text)?;
        Ok(config)
    }

    /// Loads `config.toml` next to the executable, falling back to the current
    /// directory and finally to the defaults.
    pub fn load_default() -> anyhow::Result<Self> {
        Self::load(default_config_path())
    }
}

fn default_config_path() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside_exe = dir.join("config.toml");
            if beside_exe.exists() {
                return beside_exe;
            }
        }
    }
    PathBuf::from("config.toml")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_toml(contents: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        file
    }

    #[test]
    fn missing_file_returns_defaults() {
        let config = Config::load("no/such/file.toml").unwrap();
        assert_eq!(config.ptt.key, "f12");
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.transcriber.backend, Backend::Groq);
        assert_eq!(config.cleaner.filler_patterns.len(), 8);
    }

    #[test]
    fn full_toml_overrides_everything() {
        let file = write_toml(
            r#"
            [ptt]
            key = "f9"
            [audio]
            sample_rate = 44100
            channels = 2
            min_audio_duration_ms = 500
            [whisper]
            model = "small"
            device = "cpu"
            compute_type = "int8"
            language = "en"
            [transcriber]
            max_workers = 4
            idle_unload_seconds = 600
            backend = "local"
            [cleaner]
            filler_patterns = ["\\bum\\b"]
            [writer]
            clipboard_restore_delay_ms = 100
            [groq]
            model = "whisper-large-v3-turbo"
            api_key_env = "MY_KEY"
            base_url = "https://example.com"
            "#,
        );
        let config = Config::load(file.path()).unwrap();
        assert_eq!(config.ptt.key, "f9");
        assert_eq!(config.audio.sample_rate, 44100);
        assert_eq!(config.whisper.device, "cpu");
        assert_eq!(config.transcriber.max_workers, 4);
        assert_eq!(config.transcriber.backend, Backend::Local);
        assert_eq!(config.cleaner.filler_patterns, vec!["\\bum\\b"]);
        assert_eq!(config.writer.clipboard_restore_delay_ms, 100);
        assert_eq!(config.groq.model, "whisper-large-v3-turbo");
    }

    #[test]
    fn partial_toml_merges_with_defaults() {
        let file = write_toml(
            r#"
            [ptt]
            key = "f8"
            [transcriber]
            idle_unload_seconds = 0
            "#,
        );
        let config = Config::load(file.path()).unwrap();
        assert_eq!(config.ptt.key, "f8");
        assert_eq!(config.transcriber.idle_unload_seconds, 0);
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.transcriber.max_workers, 2);
        assert_eq!(config.whisper.language, "es");
    }

    #[test]
    fn missing_sections_use_defaults() {
        let file = write_toml(
            r#"
            [audio]
            channels = 2
            "#,
        );
        let config = Config::load(file.path()).unwrap();
        assert_eq!(config.audio.channels, 2);
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.ptt.key, "f12");
        assert_eq!(config.writer.clipboard_restore_delay_ms, 50);
    }
}
