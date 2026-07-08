//! Audio recording stage.
//!
//! The `Recorder` is hardware-agnostic: it takes an injectable `AudioInput`.
//! The real implementation (`CpalInput`) drives `cpal` on a dedicated thread
//! that solely owns the `!Send` `Stream` and is controlled via commands, so the
//! `Recorder` itself can live on a Tokio task.

use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

use crate::domain::models::{AudioChunk, KeyEvent, KeyEventKind};
use crate::domain::traits::AudioInput;
use crate::utils::audio::duration_ms;

pub struct Recorder {
    sample_rate: u32,
    channels: u16,
    min_audio_duration_ms: u32,
    input: Arc<dyn AudioInput>,
    /// Pinged on every genuine key-down (not key-repeat) so the transcription
    /// backend can start reloading its model, if idle-unloaded, in parallel
    /// with the user speaking instead of after they release the key.
    warmup_tx: mpsc::Sender<()>,
    seq: u64,
    recording: bool,
}

impl Recorder {
    pub fn new(
        sample_rate: u32,
        channels: u16,
        min_audio_duration_ms: u32,
        input: Arc<dyn AudioInput>,
        warmup_tx: mpsc::Sender<()>,
    ) -> Self {
        Self {
            sample_rate,
            channels,
            min_audio_duration_ms,
            input,
            warmup_tx,
            seq: 0,
            recording: false,
        }
    }

    pub async fn run(
        &mut self,
        mut key_rx: mpsc::Receiver<KeyEvent>,
        audio_tx: mpsc::Sender<AudioChunk>,
    ) {
        while let Some(event) = key_rx.recv().await {
            match event.kind {
                KeyEventKind::Down => self.start_recording(),
                KeyEventKind::Up => self.stop_recording(&audio_tx).await,
            }
        }
    }

    fn start_recording(&mut self) {
        if self.recording {
            // OS key-repeat: ignore an extra `down` while already recording.
            tracing::debug!("Ignoring duplicate key-down while already recording (key repeat)");
            return;
        }
        // Fire-and-forget: a full channel or no receiver just means a warm-up
        // is already in flight or the backend does not need one (e.g. Groq).
        let _ = self.warmup_tx.try_send(());
        match self.input.start(self.sample_rate, self.channels) {
            Ok(()) => self.recording = true,
            Err(error) => tracing::error!(%error, "failed to start audio capture"),
        }
    }

    async fn stop_recording(&mut self, audio_tx: &mpsc::Sender<AudioChunk>) {
        if !self.recording {
            tracing::debug!("Ignoring key-up while not recording");
            return;
        }
        self.recording = false;

        // `stop()` blocks synchronously on a reply from the audio thread
        // (denoise + resample run there); run it on a blocking-pool thread so
        // it never ties up a Tokio worker thread while it waits.
        let input = self.input.clone();
        let result = tokio::task::spawn_blocking(move || input.stop()).await;
        let samples = match result {
            Ok(Ok(samples)) => samples,
            Ok(Err(error)) => {
                tracing::error!(%error, "failed to stop audio capture");
                return;
            }
            Err(error) => {
                tracing::error!(%error, "audio stop task panicked");
                return;
            }
        };

        if duration_ms(&samples, self.sample_rate) < self.min_audio_duration_ms as f64 {
            tracing::info!("Discarding recording shorter than min_audio_duration_ms");
            return;
        }

        let seq = self.seq;
        self.seq += 1;
        let _ = audio_tx
            .send(AudioChunk {
                seq,
                data: samples,
                sample_rate: self.sample_rate,
            })
            .await;
    }
}

// --- Real cpal-backed implementation ---

enum Command {
    /// `sample_rate` is the target (16 kHz); the mic opens in its native config
    /// and is resampled to this rate. Output is always mono (downmixed), so no
    /// target channel count is carried.
    Start {
        sample_rate: u32,
    },
    Stop(std::sync::mpsc::Sender<Vec<f32>>),
}

/// Real audio source. Delegates to a dedicated thread that owns the `cpal::Stream`.
pub struct CpalInput {
    tx: std::sync::mpsc::Sender<Command>,
}

impl CpalInput {
    /// `denoise`: apply RNNoise suppression when the key is released.
    pub fn new(denoise: bool) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Command>();
        std::thread::spawn(move || audio_thread(rx, denoise));
        Self { tx }
    }
}

impl Default for CpalInput {
    fn default() -> Self {
        Self::new(true)
    }
}

impl AudioInput for CpalInput {
    fn start(&self, sample_rate: u32, _channels: u16) -> anyhow::Result<()> {
        // Output is always mono, so the target channel count is ignored; the
        // downmix happens on the thread based on the native channel count.
        self.tx
            .send(Command::Start { sample_rate })
            .map_err(|_| anyhow::anyhow!("audio thread is not running"))?;
        Ok(())
    }

    fn stop(&self) -> anyhow::Result<Vec<f32>> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.tx
            .send(Command::Stop(reply_tx))
            .map_err(|_| anyhow::anyhow!("audio thread is not running"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("audio thread dropped the reply"))
    }
}

/// Audio thread loop: the sole owner of the `cpal::Stream`.
///
/// The microphone opens in its **native configuration** (on Windows/WASAPI the
/// device is often fixed to e.g. 48 kHz stereo f32 and rejects 16 kHz mono
/// requests). Samples accumulate interleaved as f32 and, on stop, are downmixed
/// to mono and resampled to the target `sample_rate` the pipeline requests.
fn audio_thread(rx: std::sync::mpsc::Receiver<Command>, denoise: bool) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    use crate::utils::audio::{denoise_48k_mono, downmix_to_mono, resample_linear};

    /// RNNoise was trained at 48 kHz; denoising always runs at this rate.
    const RNNOISE_RATE: u32 = 48_000;

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let mut stream: Option<cpal::Stream> = None;
    // Native config the stream was opened with, plus the target from Start.
    let mut native_rate: u32 = 0;
    let mut native_channels: u16 = 1;
    let mut target_rate: u32 = 16_000;

    while let Ok(command) = rx.recv() {
        match command {
            Command::Start { sample_rate } => {
                buffer.lock().unwrap().clear();
                target_rate = sample_rate;
                let device = match cpal::default_host().default_input_device() {
                    Some(device) => device,
                    None => {
                        tracing::error!("no default input device available");
                        continue;
                    }
                };
                // Native config supported by the device (avoids the
                // "stream configuration is not supported by the device" error).
                let supported = match device.default_input_config() {
                    Ok(config) => config,
                    Err(error) => {
                        tracing::error!(%error, "no default input config for device");
                        continue;
                    }
                };
                let sample_format = supported.sample_format();
                let config: cpal::StreamConfig = supported.into();
                native_rate = config.sample_rate.0;
                native_channels = config.channels;
                tracing::debug!(
                    native_rate,
                    native_channels,
                    ?sample_format,
                    target_rate,
                    "opening input stream in native config"
                );

                let built = build_capture_stream(&device, &config, sample_format, buffer.clone());
                match built {
                    Ok(started) => match started.play() {
                        Ok(()) => stream = Some(started),
                        Err(error) => tracing::error!(%error, "failed to play input stream"),
                    },
                    Err(error) => tracing::error!(%error, "failed to build input stream"),
                }
            }
            Command::Stop(reply) => {
                drop(stream.take()); // dropping the stream stops capture
                let interleaved = std::mem::take(&mut *buffer.lock().unwrap());
                let mono = downmix_to_mono(&interleaved, native_channels);
                let out = if denoise {
                    // RNNoise runs at 48 kHz: resample up, clean, then down to
                    // the target. When the mic is already 48 kHz the first
                    // resample is a cheap copy.
                    let at_48k = resample_linear(&mono, native_rate, RNNOISE_RATE);
                    let clean = denoise_48k_mono(&at_48k);
                    resample_linear(&clean, RNNOISE_RATE, target_rate)
                } else {
                    resample_linear(&mono, native_rate, target_rate)
                };
                let _ = reply.send(out);
            }
        }
    }
}

/// Builds the capture stream, converting the device's native sample format
/// (f32/i16/u16) into interleaved `f32` in `buffer`.
fn build_capture_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    buffer: Arc<Mutex<Vec<f32>>>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    use cpal::traits::DeviceTrait;

    let err_fn = |error| tracing::warn!(%error, "audio input error");
    match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                buffer.lock().unwrap().extend_from_slice(data);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mut buf = buffer.lock().unwrap();
                buf.extend(data.iter().map(|&s| s as f32 / 32768.0));
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let mut buf = buffer.lock().unwrap();
                buf.extend(data.iter().map(|&s| (s as f32 - 32768.0) / 32768.0));
            },
            err_fn,
            None,
        ),
        other => {
            tracing::error!(?other, "unsupported input sample format");
            Err(cpal::BuildStreamError::StreamConfigNotSupported)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeInput {
        samples: Vec<f32>,
        starts: Arc<AtomicUsize>,
    }

    impl AudioInput for FakeInput {
        fn start(&self, _: u32, _: u16) -> anyhow::Result<()> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn stop(&self) -> anyhow::Result<Vec<f32>> {
            Ok(self.samples.clone())
        }
    }

    fn recorder(samples: Vec<f32>, starts: Arc<AtomicUsize>) -> Recorder {
        let (warmup_tx, _warmup_rx) = mpsc::channel(1);
        Recorder::new(
            16000,
            1,
            300,
            Arc::new(FakeInput { samples, starts }),
            warmup_tx,
        )
    }

    fn down() -> KeyEvent {
        KeyEvent {
            kind: KeyEventKind::Down,
            key: "f12".into(),
        }
    }
    fn up() -> KeyEvent {
        KeyEvent {
            kind: KeyEventKind::Up,
            key: "f12".into(),
        }
    }

    async fn run_events(mut rec: Recorder, events: Vec<KeyEvent>) -> Vec<AudioChunk> {
        let (ktx, krx) = mpsc::channel(16);
        let (atx, mut arx) = mpsc::channel(16);
        for e in events {
            ktx.send(e).await.unwrap();
        }
        drop(ktx);
        rec.run(krx, atx).await;
        let mut out = Vec::new();
        while let Ok(chunk) = arx.try_recv() {
            out.push(chunk);
        }
        out
    }

    #[tokio::test]
    async fn seq_increments_per_recording() {
        let starts = Arc::new(AtomicUsize::new(0));
        let rec = recorder(vec![0.0; 16000], starts.clone());
        let out = run_events(rec, vec![down(), up(), down(), up()]).await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].seq, 0);
        assert_eq!(out[1].seq, 1);
    }

    #[tokio::test]
    async fn discards_recording_shorter_than_minimum() {
        let starts = Arc::new(AtomicUsize::new(0));
        // 160 samples at 16 kHz = 10 ms < 300 ms.
        let rec = recorder(vec![0.0; 160], starts.clone());
        let out = run_events(rec, vec![down(), up()]).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn ignores_duplicate_key_down() {
        let starts = Arc::new(AtomicUsize::new(0));
        let rec = recorder(vec![0.0; 16000], starts.clone());
        let out = run_events(rec, vec![down(), down(), up()]).await;
        assert_eq!(out.len(), 1);
        // Capture started only once despite the two `down` events.
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn ignores_key_up_without_down() {
        let starts = Arc::new(AtomicUsize::new(0));
        let rec = recorder(vec![0.0; 16000], starts.clone());
        let out = run_events(rec, vec![up()]).await;
        assert!(out.is_empty());
        assert_eq!(starts.load(Ordering::SeqCst), 0);
    }
}
