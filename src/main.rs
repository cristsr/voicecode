//! Entry point with a system tray icon.
//!
//! The pipeline (Tokio) runs on its own thread via [`Worker`]; the tray event
//! loop runs on the main thread (required on Windows/macOS). Menu: *Reiniciar
//! pipeline* (restart) and *Salir* (quit).

// In release builds this is a windowless tray app. In debug builds the
// console is kept so logs are visible on stdout.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{Icon, TrayIconBuilder};

use voicecode::Worker;

fn init_logging() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if cfg!(debug_assertions) {
        fmt().with_env_filter(filter).init();
    } else {
        // Packaged release build: log to a file next to the exe, since the
        // windowless subsystem has no stdout/stderr.
        let dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let appender = tracing_appender::rolling::never(dir, "voicecode.log");
        let (writer, guard) = tracing_appender::non_blocking(appender);
        // The guard must live for the whole program or logs get dropped.
        std::mem::forget(guard);
        fmt()
            .with_env_filter(filter)
            .with_ansi(false)
            .with_writer(writer)
            .init();
    }
}

/// Builds the tray icon. Same pixel generation as the packaged .exe's
/// resource icon (`build.rs`), so both show the same badge.
fn tray_icon() -> Icon {
    const SIZE: u32 = 64;
    let rgba = voicecode::utils::icon::render_rgba(SIZE);
    Icon::from_rgba(rgba, SIZE, SIZE).expect("valid icon rgba")
}

fn main() {
    init_logging();

    let mut worker = Worker::start();

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
                worker.restart();
            } else if event.id == quit_id {
                tracing::info!("Exit requested from tray menu");
                worker.stop();
                *control_flow = ControlFlow::Exit;
            }
        }
    });
}
