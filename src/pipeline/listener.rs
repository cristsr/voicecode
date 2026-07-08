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
        _ => None,
    }
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
    fn rejects_unknown_key() {
        assert!(parse_key("banana").is_none());
        assert!(PttListener::new("banana").is_err());
    }
}
