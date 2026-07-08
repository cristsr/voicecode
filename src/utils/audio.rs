//! Utilidades de audio (== `utils/audio.py`).

/// Concatena bloques de muestras en un único vector. `[]` si no hay bloques.
pub fn concatenate_chunks(chunks: &[Vec<f32>]) -> Vec<f32> {
    chunks.iter().flatten().copied().collect()
}

/// Duración en milisegundos de un buffer de muestras a `sample_rate`.
pub fn duration_ms(samples: &[f32], sample_rate: u32) -> f64 {
    (samples.len() as f64 / sample_rate as f64) * 1000.0
}

/// Convierte muestras intercaladas de `channels` canales a mono promediando cada
/// frame. Con `channels <= 1` devuelve una copia. Esto reemplaza el "tomar solo
/// el canal 0": promediar conserva mejor la señal si el micro entrega estéreo.
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

/// Resamplea audio mono de `from_rate` a `to_rate` por interpolación lineal.
/// Suficiente para voz + Whisper (robusto a artefactos menores). Si las tasas
/// coinciden o la entrada está vacía, devuelve una copia sin tocar.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        // Posición equivalente en la señal de entrada.
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac as f32);
    }
    out
}

/// Supresión de ruido con **RNNoise** (`nnnoiseless`). Recibe audio **mono a
/// 48 kHz** (la tasa con la que RNNoise fue entrenado) y devuelve el mismo audio
/// con el ruido de fondo atenuado, conservando la voz.
///
/// Cómo funciona RNNoise, en breve: procesa el audio en tramas de 10 ms (480
/// muestras a 48 kHz). Por cada trama calcula una representación espectral (bandas
/// de frecuencia estilo Bark) y una pequeña red neuronal recurrente (GRU) —
/// entrenada con voz limpia + ruido — estima, banda por banda, **cuánta señal es
/// voz vs. ruido**. Con eso aplica una ganancia por banda: deja pasar las bandas
/// dominadas por voz y atenúa las dominadas por ruido. Al ser recurrente, "recuerda"
/// el contexto reciente, así distingue ruido estacionario (ventilador, hiss) de la
/// voz. No separa hablantes: si el ruido *es* otra voz, tiende a conservarla.
///
/// `nnnoiseless` espera muestras en rango de `i16`, no `[-1, 1]`; por eso se escala
/// al entrar y se desescala al salir. La longitud de salida iguala a la de entrada.
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
        // Escala [-1, 1] -> rango i16 y rellena con ceros la última trama parcial.
        for (i, slot) in in_frame.iter_mut().enumerate() {
            *slot = chunk.get(i).copied().unwrap_or(0.0) * 32768.0;
        }
        state.process_frame(&mut out_frame, &in_frame);
        // Desescala y recorta al largo real de la trama (la última puede ser < 480).
        out.extend(out_frame.iter().take(chunk.len()).map(|&s| s / 32768.0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concatenate_empty_is_empty() {
        let empty: Vec<Vec<f32>> = vec![];
        assert!(concatenate_chunks(&empty).is_empty());
    }

    #[test]
    fn concatenate_joins_in_order() {
        let chunks = vec![vec![1.0, 2.0], vec![3.0]];
        assert_eq!(concatenate_chunks(&chunks), vec![1.0, 2.0, 3.0]);
    }

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
        // Frames: (0.0, 1.0), (0.5, 0.5) -> 0.5, 0.5
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
        // 48 kHz -> 16 kHz: 1/3 de las muestras aprox.
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
        // 1000 muestras = 2 tramas completas (960) + 1 parcial (40).
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
