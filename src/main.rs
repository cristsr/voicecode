//! Entry point con ícono de bandeja (== `tray_app.py`).
//!
//! El pipeline (Tokio) corre en un thread aparte; el event loop del tray corre
//! en el thread principal (requisito en Windows/macOS). Menú: *Reiniciar
//! pipeline* y *Salir*.

// En release (producción) es una app de bandeja sin ventana de consola. En debug
// se mantiene la consola para ver los logs por stdout (== logging a consola).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::thread::JoinHandle;

use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tokio_util::sync::CancellationToken;
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{Icon, TrayIconBuilder};

use voicecode::config::Config;
use voicecode::run_pipeline;

/// Arranca el pipeline en su propio runtime de Tokio, en un thread aparte.
/// Devuelve un handle para cancelarlo (== el thread con event loop de `tray_app.py`).
struct Pipeline {
    cancel: CancellationToken,
    handle: Option<JoinHandle<()>>,
}

impl Pipeline {
    fn start() -> Self {
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let handle = std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(error) => {
                    tracing::error!(%error, "failed to build Tokio runtime");
                    return;
                }
            };
            runtime.block_on(async move {
                let config = match Config::load_default() {
                    Ok(config) => config,
                    Err(error) => {
                        tracing::error!(%error, "failed to load config");
                        return;
                    }
                };
                tokio::select! {
                    result = run_pipeline(config) => {
                        if let Err(error) = result {
                            tracing::error!(%error, "pipeline crashed");
                        }
                    }
                    _ = token.cancelled() => tracing::info!("pipeline cancelled"),
                }
            });
        });
        Self {
            cancel,
            handle: Some(handle),
        }
    }

    fn stop(&mut self) {
        self.cancel.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn init_logging() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if cfg!(debug_assertions) {
        fmt().with_env_filter(filter).init();
    } else {
        // Build de release (empaquetado): log a un archivo junto al exe, ya que
        // con `--noconsole` no hay stdout/stderr (== `_configure_logging`).
        let dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let appender = tracing_appender::rolling::never(dir, "voicecode.log");
        let (writer, guard) = tracing_appender::non_blocking(appender);
        // El guard debe vivir todo el programa para no perder logs.
        std::mem::forget(guard);
        fmt()
            .with_env_filter(filter)
            .with_ansi(false)
            .with_writer(writer)
            .init();
    }
}

/// Ícono azul sólido 64x64 (placeholder; == el ícono dibujado con PIL).
fn tray_icon() -> Icon {
    const SIZE: u32 = 64;
    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for _ in 0..(SIZE * SIZE) {
        rgba.extend_from_slice(&[74, 144, 217, 255]);
    }
    Icon::from_rgba(rgba, SIZE, SIZE).expect("valid icon rgba")
}

fn main() {
    init_logging();

    let mut pipeline = Pipeline::start();

    let menu = Menu::new();
    let restart_item = MenuItem::new("Reiniciar pipeline", true, None);
    let quit_item = MenuItem::new("Salir", true, None);
    menu.append(&restart_item).expect("append restart item");
    menu.append(&quit_item).expect("append quit item");

    let _tray = TrayIconBuilder::new()
        .with_tooltip("VoiceCode")
        .with_icon(tray_icon())
        .with_menu(Box::new(menu))
        .build()
        .expect("build tray icon");

    let menu_channel = MenuEvent::receiver();
    let restart_id = restart_item.id().clone();
    let quit_id = quit_item.id().clone();

    let event_loop = EventLoopBuilder::new().build();
    event_loop.run(move |_event, _target, control_flow| {
        *control_flow = ControlFlow::Wait;

        while let Ok(event) = menu_channel.try_recv() {
            if event.id == restart_id {
                tracing::info!("Restart requested from tray menu");
                pipeline.stop();
                pipeline = Pipeline::start();
            } else if event.id == quit_id {
                tracing::info!("Exit requested from tray menu");
                pipeline.stop();
                *control_flow = ControlFlow::Exit;
            }
        }
    });
}
