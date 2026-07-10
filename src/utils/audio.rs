//! Audio helpers: duration, downmixing, resampling, gain and noise suppression.

/// Duration in milliseconds of a sample buffer at `sample_rate`.
pub fn duration_ms(samples: &[f32], sample_rate: u32) -> f64 {
    (samples.len() as f64 / sample_rate as f64) * 1000.0
}

/// Peak amplitude the limiter keeps the normalized output below, leaving a
/// little headroom under full scale so the boosted signal never clips.
const PEAK_CEILING: f32 = 0.98;
/// Below this input RMS a take is treated as silence/background and left
/// unamplified, so hiss is not blown up to speech level.
const SILENCE_FLOOR_RMS: f32 = 1e-4;

/// Measured level of a recording and the gain the normalizer applied to it.
/// Logged per recording so the incoming audio level can be inspected.
#[derive(Debug, Clone, Copy)]
pub struct GainReport {
    pub input_rms: f32,
    pub input_peak: f32,
    pub gain: f32,
    pub output_peak: f32,
}

/// Converts a linear amplitude (0..1) to dBFS; ~0 maps to a large negative.
pub fn to_dbfs(x: f32) -> f32 {
    if x <= 1e-9 {
        -120.0
    } else {
        20.0 * x.log10()
    }
}

/// Lifts a quiet recording toward a consistent speech level before it reaches
/// Whisper, which transcribes low-level audio noticeably worse (the reason a
/// far-away mic forces the speaker to raise their voice). The gain aims the RMS
/// at `target_rms`, is capped by `max_gain` so near-silence/background hiss is
/// not amplified to speech level, and never attenuates (`gain >= 1.0`) so
/// already-loud audio is left untouched. A final peak limiter overrides the
/// target when needed so the result never clips. Pass `max_gain = 1.0` to only
/// measure the level without changing the audio.
pub fn normalize_speech_level(
    samples: &[f32],
    target_rms: f32,
    max_gain: f32,
) -> (Vec<f32>, GainReport) {
    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    let rms = if samples.is_empty() {
        0.0
    } else {
        (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    };

    let mut gain = 1.0f32;
    if rms > SILENCE_FLOOR_RMS && target_rms > 0.0 {
        gain = (target_rms / rms).clamp(1.0, max_gain.max(1.0));
        // Peak limiter takes priority over hitting the RMS target so a take
        // with sharp transients (a plosive, a tap) is never pushed into clip.
        if peak > 0.0 {
            gain = (gain).min(PEAK_CEILING / peak).max(1.0);
        }
    }

    let out = if gain == 1.0 {
        samples.to_vec()
    } else {
        samples.iter().map(|&s| s * gain).collect()
    };
    let report = GainReport {
        input_rms: rms,
        input_peak: peak,
        gain,
        output_peak: peak * gain,
    };
    (out, report)
}

/// Downmixes interleaved multi-channel samples to mono by averaging each frame.
/// Averaging (rather than picking channel 0) preserves the signal better when
/// the microphone delivers stereo. Returns a copy when already mono.
pub fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels.max(1) as usize;
    if channels == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resamples mono audio from `from_rate` to `to_rate` by linear interpolation,
/// which is good enough for speech feeding Whisper. Returns an untouched copy
/// when the rates match or the input is empty.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac as f32);
    }
    out
}

/// Suppresses background noise with RNNoise (`nnnoiseless`) while preserving
/// speech. Expects mono audio at 48 kHz, the rate RNNoise was trained on, and
/// returns a buffer of the same length. `nnnoiseless` works in `i16` range, so
/// samples are scaled on the way in and back to `[-1, 1]` on the way out.
pub fn denoise_48k_mono(samples: &[f32]) -> Vec<f32> {
    use nnnoiseless::{DenoiseState, FRAME_SIZE};

    if samples.is_empty() {
        return Vec::new();
    }
    let mut state = DenoiseState::new();
    let mut in_frame = [0.0f32; FRAME_SIZE];
    let mut out_frame = [0.0f32; FRAME_SIZE];
    let mut out = Vec::with_capacity(samples.len());

    for chunk in samples.chunks(FRAME_SIZE) {
        for (i, slot) in in_frame.iter_mut().enumerate() {
            *slot = chunk.get(i).copied().unwrap_or(0.0) * 32768.0;
        }
        state.process_frame(&mut out_frame, &in_frame);
        // Trim the trailing partial frame back to its real length.
        out.extend(out_frame.iter().take(chunk.len()).map(|&s| s / 32768.0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_ms_matches_sample_rate() {
        let samples = vec![0.0f32; 16000];
        assert_eq!(duration_ms(&samples, 16000), 1000.0);
    }

    #[test]
    fn downmix_mono_is_identity() {
        let samples = vec![0.1, 0.2, 0.3];
        assert_eq!(downmix_to_mono(&samples, 1), samples);
    }

    #[test]
    fn downmix_stereo_averages_frames() {
        let interleaved = vec![0.0, 1.0, 0.5, 0.5];
        assert_eq!(downmix_to_mono(&interleaved, 2), vec![0.5, 0.5]);
    }

    #[test]
    fn resample_same_rate_is_identity() {
        let samples = vec![1.0, 2.0, 3.0];
        assert_eq!(resample_linear(&samples, 16000, 16000), samples);
    }

    #[test]
    fn resample_halving_rate_reduces_length() {
        let samples = vec![0.0f32; 48000];
        let out = resample_linear(&samples, 48000, 16000);
        assert_eq!(out.len(), 16000);
    }

    #[test]
    fn resample_preserves_constant_signal() {
        let samples = vec![0.7f32; 4800];
        let out = resample_linear(&samples, 48000, 16000);
        assert!(out.iter().all(|&s| (s - 0.7).abs() < 1e-6));
    }

    #[test]
    fn denoise_preserves_length_and_handles_partial_frame() {
        let samples = vec![0.1f32; 1000];
        let out = denoise_48k_mono(&samples);
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn denoise_empty_is_empty() {
        assert!(denoise_48k_mono(&[]).is_empty());
    }

    #[test]
    fn denoise_silence_stays_silent() {
        let out = denoise_48k_mono(&vec![0.0f32; 960]);
        assert!(out.iter().all(|&s| s.abs() < 1e-3));
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn normalize_boosts_quiet_speech_toward_target() {
        let quiet = vec![0.02f32, -0.02, 0.02, -0.02];
        let (out, report) = normalize_speech_level(&quiet, 0.12, 8.0);
        assert!(report.gain > 1.0);
        // Output level moved up toward the target without clipping.
        assert!(rms(&out) > rms(&quiet));
        assert!(report.output_peak <= PEAK_CEILING + 1e-6);
    }

    #[test]
    fn normalize_leaves_loud_audio_untouched() {
        // RMS 0.5 already above target: never attenuate.
        let loud = vec![0.5f32, -0.5, 0.5, -0.5];
        let (out, report) = normalize_speech_level(&loud, 0.12, 8.0);
        assert_eq!(report.gain, 1.0);
        assert_eq!(out, loud);
    }

    #[test]
    fn normalize_never_clips_even_with_aggressive_target() {
        let sig = vec![0.3f32, -0.3, 0.3, -0.3];
        let (_out, report) = normalize_speech_level(&sig, 0.99, 8.0);
        assert!(report.output_peak <= PEAK_CEILING + 1e-6);
    }

    #[test]
    fn normalize_leaves_silence_unamplified() {
        let (out, report) = normalize_speech_level(&vec![0.0f32; 100], 0.12, 8.0);
        assert_eq!(report.gain, 1.0);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn normalize_with_unit_max_gain_only_measures() {
        let quiet = vec![0.02f32, -0.02, 0.02, -0.02];
        let (out, report) = normalize_speech_level(&quiet, 0.12, 1.0);
        assert_eq!(report.gain, 1.0);
        assert_eq!(out, quiet);
        assert!((report.input_rms - 0.02).abs() < 1e-6);
    }
}
