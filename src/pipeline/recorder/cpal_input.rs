//! Real `cpal`-backed audio source.
//!
//! `CpalInput` delegates to a dedicated thread that solely owns the `!Send`
//! `cpal::Stream` and is driven by `Command`s, so the `Recorder` above it can
//! live on a Tokio task. Downmixing, denoising, resampling and level
//! normalization all run on this thread when a recording stops.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::domain::traits::AudioInput;

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
    /// `auto_gain`: normalize the recording's level toward a consistent speech
    /// loudness before it leaves the audio thread.
    pub fn new(denoise: bool, auto_gain: bool) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Command>();
        std::thread::spawn(move || audio_thread(rx, denoise, auto_gain));
        Self { tx }
    }
}

impl Default for CpalInput {
    fn default() -> Self {
        Self::new(true, true)
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
///
/// The stream is opened **once** and kept playing for the life of the process,
/// rather than being reopened on every key-down. On Windows/WASAPI the audio
/// endpoint drops into a low-power state after a boot or a long idle, and the
/// first activation then loses the first few hundred ms of audio — which made
/// the first utterance after an idle period come in under
/// `min_audio_duration_ms` and get silently discarded. Keeping the stream alive
/// holds the endpoint warm; the `capturing` flag gates whether callback samples
/// are accumulated (during a recording) or dropped (while idle).
fn audio_thread(rx: std::sync::mpsc::Receiver<Command>, denoise: bool, auto_gain: bool) {
    use crate::utils::audio::{
        denoise_48k_mono, downmix_to_mono, normalize_speech_level, resample_linear, to_dbfs,
    };

    /// RNNoise was trained at 48 kHz; denoising always runs at this rate.
    const RNNOISE_RATE: u32 = 48_000;
    /// Target RMS (~ -18 dBFS) the auto-gain lifts quiet speech toward.
    const AUTO_GAIN_TARGET_RMS: f32 = 0.12;
    /// Ceiling on the boost so background hiss on a near-silent take is not
    /// amplified to speech level.
    const AUTO_GAIN_MAX: f32 = 8.0;

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    // Gates the capture callback: samples are accumulated only between a Start
    // and its Stop. While `false` the stream keeps running (holding the WASAPI
    // endpoint warm) but its samples are dropped.
    let capturing = Arc::new(AtomicBool::new(false));
    let mut stream: Option<cpal::Stream> = None;
    // Native config the stream was opened with, plus the target from Start.
    let mut native_rate: u32 = 0;
    let mut native_channels: u16 = 1;
    let mut target_rate: u32 = 16_000;

    // Open the stream up front so the endpoint is already warm by the first
    // recording. If no device is available yet, this is retried on Start.
    open_stream(
        &mut stream,
        &mut native_rate,
        &mut native_channels,
        &buffer,
        &capturing,
    );

    while let Ok(command) = rx.recv() {
        match command {
            Command::Start { sample_rate } => {
                target_rate = sample_rate;
                buffer.lock().unwrap().clear();
                if stream.is_none() {
                    open_stream(
                        &mut stream,
                        &mut native_rate,
                        &mut native_channels,
                        &buffer,
                        &capturing,
                    );
                }
                if stream.is_some() {
                    capturing.store(true, Ordering::Release);
                }
            }
            Command::Stop(reply) => {
                capturing.store(false, Ordering::Release);
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
                // Measure the incoming level (and boost it when enabled). With
                // `max_gain = 1.0` this only measures, leaving the audio as-is.
                let max_gain = if auto_gain { AUTO_GAIN_MAX } else { 1.0 };
                let (out, level) = normalize_speech_level(&out, AUTO_GAIN_TARGET_RMS, max_gain);
                tracing::info!(
                    rms_dbfs = to_dbfs(level.input_rms),
                    peak_dbfs = to_dbfs(level.input_peak),
                    gain_db = to_dbfs(level.gain),
                    out_peak = level.output_peak,
                    "input audio level"
                );
                let _ = reply.send(out);
            }
        }
    }
}

/// Opens the default input device in its native config and starts it playing,
/// storing the stream and its native rate/channels. The capture callback only
/// accumulates into `buffer` while `capturing` is set. No-op logging on failure
/// so a missing device at startup is retried on the next Start.
fn open_stream(
    stream: &mut Option<cpal::Stream>,
    native_rate: &mut u32,
    native_channels: &mut u16,
    buffer: &Arc<Mutex<Vec<f32>>>,
    capturing: &Arc<AtomicBool>,
) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let device = match cpal::default_host().default_input_device() {
        Some(device) => device,
        None => {
            tracing::error!("no default input device available");
            return;
        }
    };
    // Native config supported by the device (avoids the
    // "stream configuration is not supported by the device" error).
    let supported = match device.default_input_config() {
        Ok(config) => config,
        Err(error) => {
            tracing::error!(%error, "no default input config for device");
            return;
        }
    };
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    *native_rate = config.sample_rate.0;
    *native_channels = config.channels;
    tracing::debug!(
        native_rate = *native_rate,
        native_channels = *native_channels,
        ?sample_format,
        "opening input stream in native config (kept warm)"
    );

    let built = build_capture_stream(
        &device,
        &config,
        sample_format,
        buffer.clone(),
        capturing.clone(),
    );
    match built {
        Ok(started) => match started.play() {
            Ok(()) => *stream = Some(started),
            Err(error) => tracing::error!(%error, "failed to play input stream"),
        },
        Err(error) => tracing::error!(%error, "failed to build input stream"),
    }
}

/// Builds the capture stream, converting the device's native sample format
/// (f32/i16/u16) into interleaved `f32` in `buffer`. The stream runs
/// continuously to keep the endpoint warm; callbacks only accumulate samples
/// while `capturing` is set (i.e. between a Start and its Stop).
fn build_capture_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    buffer: Arc<Mutex<Vec<f32>>>,
    capturing: Arc<AtomicBool>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    use cpal::traits::DeviceTrait;

    let err_fn = |error| tracing::warn!(%error, "audio input error");
    match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !capturing.load(Ordering::Acquire) {
                    return;
                }
                buffer.lock().unwrap().extend_from_slice(data);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                if !capturing.load(Ordering::Acquire) {
                    return;
                }
                let mut buf = buffer.lock().unwrap();
                buf.extend(data.iter().map(|&s| s as f32 / 32768.0));
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                if !capturing.load(Ordering::Acquire) {
                    return;
                }
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
