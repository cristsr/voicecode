//! Global push-to-talk key listener.
//!
//! `rdev::listen` blocks the thread, so it runs on a dedicated `std::thread`
//! that forwards events over an `mpsc` channel.

use rdev::{Event, EventType, Key};
use tokio::sync::mpsc;

use crate::domain::models::{KeyEvent, KeyEventKind};

pub struct PttListener {
    key_name: String,
    target: Key,
}

impl PttListener {
    /// Creates the listener, validating the configured PTT key name.
    pub fn new(key_name: &str) -> anyhow::Result<Self> {
        let target = parse_key(key_name)
            .ok_or_else(|| anyhow::anyhow!("unsupported PTT key: '{key_name}'"))?;
        Ok(Self {
            key_name: key_name.to_string(),
            target,
        })
    }

    /// Starts the global listener. Does not return while the pipeline is alive.
    pub async fn run(&self, key_tx: mpsc::Sender<KeyEvent>) {
        let target = self.target;
        let key_name = self.key_name.clone();

        std::thread::spawn(move || {
            let callback = move |event: Event| {
                let kind = match event.event_type {
                    EventType::KeyPress(k) if k == target => Some(KeyEventKind::Down),
                    EventType::KeyRelease(k) if k == target => Some(KeyEventKind::Up),
                    _ => None,
                };
                if let Some(kind) = kind {
                    let msg = KeyEvent {
                        kind,
                        key: key_name.clone(),
                    };
                    // try_send: never block the OS keyboard hook. PTT events
                    // are rare, so the channel does not fill up in practice.
                    if let Err(error) = key_tx.try_send(msg) {
                        tracing::warn!(%error, "dropped PTT key event");
                    }
                }
            };
            if let Err(error) = rdev::listen(callback) {
                tracing::error!(?error, "keyboard listener stopped");
            }
        });

        // Keep this stage alive for as long as the pipeline runs.
        std::future::pending::<()>().await;
    }
}

/// Maps a configured key name to its `rdev::Key`.
fn parse_key(name: &str) -> Option<Key> {
    match name.to_ascii_lowercase().as_str() {
        "f1" => Some(Key::F1),
        "f2" => Some(Key::F2),
        "f3" => Some(Key::F3),
        "f4" => Some(Key::F4),
        "f5" => Some(Key::F5),
        "f6" => Some(Key::F6),
        "f7" => Some(Key::F7),
        "f8" => Some(Key::F8),
        "f9" => Some(Key::F9),
        "f10" => Some(Key::F10),
        "f11" => Some(Key::F11),
        "f12" => Some(Key::F12),
        "space" => Some(Key::Space),
        "ctrl_r" | "control_r" => Some(Key::ControlRight),
        "alt_r" => Some(Key::AltGr),
        other => parse_extended_function_key(other),
    }
}

/// F13-F24: `rdev` has no named variants for these (absent on common keyboards,
/// but emitted by programmable ones like a Corne with remapped layers). `rdev`
/// still receives them as `Key::Unknown(code)`; this names them using each
/// platform's native key code. Verified against `rdev`'s own source
/// (`windows/keycodes.rs` and `linux/keycodes.rs`); the Windows case was also
/// confirmed on real hardware (Corne/QMK keyboard) on 2026-07-07.
#[cfg(windows)]
fn parse_extended_function_key(name: &str) -> Option<Key> {
    // VK_F13..VK_F24 (Win32 virtual-key codes).
    let code: u32 = match name {
        "f13" => 0x7C,
        "f14" => 0x7D,
        "f15" => 0x7E,
        "f16" => 0x7F,
        "f17" => 0x80,
        "f18" => 0x81,
        "f19" => 0x82,
        "f20" => 0x83,
        "f21" => 0x84,
        "f22" => 0x85,
        "f23" => 0x86,
        "f24" => 0x87,
        _ => return None,
    };
    Some(Key::Unknown(code))
}

/// `rdev` 0.5's Linux backend only supports X11 (XRecord via Xlib) — no native
/// Wayland backend. Under pure Wayland (no XWayland, or a compositor that does
/// not expose XRecord) `rdev::listen` may capture nothing regardless of the
/// configured key; that is a separate limitation, not a problem with this
/// table. Codes = evdev keycode (`linux/input-event-codes.h`, KEY_F13..KEY_F24
/// = 183..194) + 8, the standard X11/XKB keycode convention (also used by
/// Wayland via xkbcommon). Not verified on real hardware: no Linux environment
/// was available to test this.
#[cfg(target_os = "linux")]
fn parse_extended_function_key(name: &str) -> Option<Key> {
    let code: u32 = match name {
        "f13" => 191,
        "f14" => 192,
        "f15" => 193,
        "f16" => 194,
        "f17" => 195,
        "f18" => 196,
        "f19" => 197,
        "f20" => 198,
        "f21" => 199,
        "f22" => 200,
        "f23" => 201,
        "f24" => 202,
        _ => return None,
    };
    Some(Key::Unknown(code))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn parse_extended_function_key(_name: &str) -> Option<Key> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_keys_case_insensitively() {
        assert_eq!(parse_key("f12"), Some(Key::F12));
        assert_eq!(parse_key("F9"), Some(Key::F9));
        assert_eq!(parse_key("space"), Some(Key::Space));
    }

    #[test]
    #[cfg(windows)]
    fn parses_extended_function_keys_via_windows_vk_codes() {
        assert_eq!(parse_key("f13"), Some(Key::Unknown(0x7C)));
        assert_eq!(parse_key("F13"), Some(Key::Unknown(0x7C)));
        assert_eq!(parse_key("f24"), Some(Key::Unknown(0x87)));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn parses_extended_function_keys_via_linux_x11_keycodes() {
        assert_eq!(parse_key("f13"), Some(Key::Unknown(191)));
        assert_eq!(parse_key("F13"), Some(Key::Unknown(191)));
        assert_eq!(parse_key("f24"), Some(Key::Unknown(202)));
    }

    #[test]
    fn rejects_unknown_key() {
        assert!(parse_key("banana").is_none());
        assert!(PttListener::new("banana").is_err());
    }
}
