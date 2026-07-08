//! Audio helpers: duration, downmixing, resampling and noise suppression.

/// Duration in milliseconds of a sample buffer at `sample_rate`.
pub fn duration_ms(samples: &[f32], sample_rate: u32) -> f64 {
    (samples.len() as f64 / sample_rate as f64) * 1000.0
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
}
