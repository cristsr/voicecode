//! Escucha global de la tecla PTT (== `pipeline/listener.py`).
//!
//! `rdev::listen` bloquea el thread, así que corre en un `std::thread` dedicado
//! que reenvía los eventos por un canal `mpsc` (== el bridge `pynput` →
//! `call_soon_threadsafe` del proyecto Python).

use rdev::{Event, EventType, Key};
use tokio::sync::mpsc;

use crate::domain::models::{KeyEvent, KeyEventKind};

pub struct PttListener {
    key_name: String,
    target: Key,
}

impl PttListener {
    /// Crea el listener validando el nombre de la tecla PTT.
    pub fn new(key_name: &str) -> anyhow::Result<Self> {
        let target = parse_key(key_name)
            .ok_or_else(|| anyhow::anyhow!("tecla PTT no soportada: '{key_name}'"))?;
        Ok(Self {
            key_name: key_name.to_string(),
            target,
        })
    }

    /// Arranca la escucha global. No retorna mientras el pipeline viva.
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
                    // try_send: no bloquear el hook de teclado del OS. Los eventos
                    // PTT son escasos, el canal no se llena en la práctica.
                    if let Err(error) = key_tx.try_send(msg) {
                        tracing::warn!(%error, "dropped PTT key event");
                    }
                }
            };
            if let Err(error) = rdev::listen(callback) {
                tracing::error!(?error, "keyboard listener stopped");
            }
        });

        // Mantener viva la etapa mientras el pipeline corre (== await Event().wait()).
        std::future::pending::<()>().await;
    }
}

/// Mapea un nombre de tecla de config a la `rdev::Key` correspondiente.
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

/// F13–F24: `rdev` no tiene variantes nombradas para estas (no existen en
/// teclados comunes, pero sí las emiten teclados programables como un Corne con
/// capas remapeadas). rdev igual las recibe como `Key::Unknown(código)`; acá les
/// damos nombre usando el código de tecla virtual de Windows (VK_F13..VK_F24 =
/// 0x7C..0x87). Esos códigos son específicos de Windows: en Linux (evdev) estos
/// mismos nombres necesitarían otra tabla de códigos.
#[cfg(windows)]
fn parse_extended_function_key(name: &str) -> Option<Key> {
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

#[cfg(not(windows))]
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
    fn rejects_unknown_key() {
        assert!(parse_key("banana").is_none());
        assert!(PttListener::new("banana").is_err());
    }
}
