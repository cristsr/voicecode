//! Modelos de dominio. Equivalen a los `dataclass(frozen=True)` de
//! `domain/models.py`: inmutables por convención (nunca se mutan tras crearse).

/// Tipo de evento de tecla del push-to-talk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyEventKind {
    Down,
    Up,
}

/// Evento de la tecla PTT (`== KeyEvent` en Python).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyEvent {
    pub kind: KeyEventKind,
    pub key: String,
}

/// Bloque de audio grabado, con su número de secuencia de origen.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioChunk {
    pub seq: u64,
    pub data: Vec<f32>,
    pub sample_rate: u32,
}

/// Texto crudo devuelto por el transcriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscribedText {
    pub seq: u64,
    pub raw: String,
}

/// Texto ya limpiado, listo para pegar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CleanText {
    pub seq: u64,
    pub text: String,
}
