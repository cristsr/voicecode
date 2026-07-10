//! VoiceCode: push-to-talk dictation. A pipeline of decoupled stages connected
//! by `tokio::sync::mpsc` channels.
//!
//! [`Pipeline`] assembles and runs the stages; [`Worker`] owns the process-level
//! lifecycle (thread, runtime, restart) that the tray UI drives.

pub mod config;
pub mod domain;
pub mod pipeline;
pub mod utils;
pub mod worker;

pub use pipeline::Pipeline;
pub use worker::Worker;
