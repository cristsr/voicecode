//! Prueba de integración manual del backend Groq: transcribe un archivo de audio
//! y reporta el texto y la latencia real de ida y vuelta.
//!
//! Uso:
//!   $env:GROQ_API_KEY="gsk_..."      # PowerShell
//!   cargo run --example groq_check -- ruta\al\audio.wav
//!
//! Acepta WAV (16-bit int o float, mono o multicanal — se hace downmix a mono).

use std::time::Instant;

use voicecode::config::Config;
use voicecode::domain::traits::TranscriptionBackend;
use voicecode::pipeline::transcriber::groq::GroqBackend;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| {
        anyhow::anyhow!("uso: cargo run --example groq_check -- <ruta_audio.wav>")
    })?;

    let (samples, sample_rate) = read_wav_mono_f32(&path)?;
    println!(
        "Audio: {} muestras, {} Hz, {:.1}s",
        samples.len(),
        sample_rate,
        samples.len() as f32 / sample_rate as f32
    );

    let config = Config::load_default().unwrap_or_default();
    let backend = GroqBackend::from_config(&config)?;

    println!("Enviando a Groq ({})...", config.groq.model);
    let start = Instant::now();
    let text = backend
        .transcribe(&samples, sample_rate, &config.whisper.language)
        .await?;
    let elapsed = start.elapsed();

    println!("\n--- Transcripción ---\n{text}\n---------------------");
    println!(
        "Latencia total (encode + red + modelo): {:.0} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Lee un WAV a f32 mono. Downmix por promedio si hay varios canales.
fn read_wav_mono_f32(path: &str) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };

    let mono = if channels <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}
