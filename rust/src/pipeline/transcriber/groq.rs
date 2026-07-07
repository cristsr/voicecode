//! Backend de transcripción vía la API de Groq (compatible con OpenAI).
//! Encodea el audio a WAV en memoria y lo sube por multipart.

use std::io::Cursor;

use async_trait::async_trait;

use crate::config::Config;
use crate::domain::traits::TranscriptionBackend;

pub struct GroqBackend {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl GroqBackend {
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let api_key = std::env::var(&config.groq.api_key_env).map_err(|_| {
            anyhow::anyhow!(
                "falta la variable de entorno {} con la API key de Groq",
                config.groq.api_key_env
            )
        })?;
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model: config.groq.model.clone(),
            base_url: config.groq.base_url.trim_end_matches('/').to_string(),
        })
    }
}

/// Encodea PCM f32 mono [-1, 1] a un WAV de 16-bit en memoria.
fn encode_wav(audio: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
    for &sample in audio {
        let clamped = sample.clamp(-1.0, 1.0);
        writer.write_sample((clamped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(cursor.into_inner())
}

#[derive(serde::Deserialize)]
struct GroqResponse {
    text: String,
}

#[async_trait]
impl TranscriptionBackend for GroqBackend {
    async fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: &str,
    ) -> anyhow::Result<String> {
        let wav = encode_wav(audio, sample_rate)?;
        let part = reqwest::multipart::Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;
        let form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", self.model.clone())
            .text("language", language.to_string())
            .text("response_format", "json");

        let url = format!("{}/audio/transcriptions", self.base_url);
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await?
            .error_for_status()?;

        let body: GroqResponse = response.json().await?;
        Ok(body.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_wav_writes_valid_riff_header() {
        let wav = encode_wav(&[0.0, 0.5, -0.5], 16000).unwrap();
        // Cabecera RIFF/WAVE.
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        // 44 bytes de cabecera + 3 muestras * 2 bytes.
        assert_eq!(wav.len(), 44 + 3 * 2);
    }
}
