//! Domain models. Immutable by convention: never mutated after creation.

/// Kind of push-to-talk key event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyEventKind {
    Down,
    Up,
}

/// A push-to-talk key event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyEvent {
    pub kind: KeyEventKind,
    pub key: String,
}

/// A recorded audio block tagged with its source sequence number.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioChunk {
    pub seq: u64,
    pub data: Vec<f32>,
    pub sample_rate: u32,
}

/// Raw text returned by the transcriber.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscribedText {
    pub seq: u64,
    pub raw: String,
}

/// Cleaned text, ready to paste.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CleanText {
    pub seq: u64,
    pub text: String,
}
