//! Entry point with a system tray icon.
//!
//! The pipeline (Tokio) runs on its own thread; the tray event loop runs on
//! the main thread (required on Windows/macOS). Menu: *Reiniciar pipeline*
//! (restart) and *Salir* (quit).

// In release builds this is a windowless tray app. In debug builds the
// console is kept so logs are visible on stdout.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::thread::JoinHandle;

use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tokio_util::sync::CancellationToken;
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{Icon, TrayIconBuilder};

use voicecode::config::Config;
use voicecode::run_pipeline;

/// Runs the pipeline on its own Tokio runtime, on a dedicated thread, and
/// exposes a handle to cancel it.
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

/// Eases `x` from 0 to 1 as it crosses from `edge0` to `edge1`, for antialiasing.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Signed distance from `(px, py)` to a capsule (a thick line segment) between
/// `(ax, ay)` and `(bx, by)` with the given radius.
fn capsule_sdf(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32, r: f32) -> f32 {
    let (pax, pay) = (px - ax, py - ay);
    let (bax, bay) = (bx - ax, by - ay);
    let h = ((pax * bax + pay * bay) / (bax * bax + bay * bay)).clamp(0.0, 1.0);
    let (dx, dy) = (pax - bax * h, pay - bay * h);
    (dx * dx + dy * dy).sqrt() - r
}

/// Indigo circular badge with a white microphone glyph, rendered procedurally
/// (signed-distance shapes, antialiased) so no image asset needs to be bundled.
fn tray_icon() -> Icon {
    const SIZE: f32 = 64.0;
    const BG: (f32, f32, f32) = (79.0, 70.0, 229.0);
    const FG: (f32, f32, f32) = (255.0, 255.0, 255.0);

    let center = SIZE / 2.0;
    let bg_radius = SIZE * 0.47;

    // Microphone glyph: a capsule head, a "U"-shaped stand ring below it, and
    // a leg + base connecting it to the bottom of the badge.
    let body_top = center - SIZE * 0.24;
    let body_bottom = center - SIZE * 0.02;
    let body_radius = SIZE * 0.10;

    let stand_center_y = body_bottom - SIZE * 0.01;
    let stand_radius = SIZE * 0.145;
    let stand_thickness = SIZE * 0.035;

    let leg_top = stand_center_y + stand_radius - SIZE * 0.01;
    let leg_bottom = center + SIZE * 0.26;
    let leg_radius = SIZE * 0.02;

    let base_half_width = SIZE * 0.11;
    let base_radius = SIZE * 0.02;

    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4.0) as usize);
    for y in 0..SIZE as u32 {
        for x in 0..SIZE as u32 {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            let dist_bg = ((px - center).powi(2) + (py - center).powi(2)).sqrt() - bg_radius;
            let alpha_bg = 1.0 - smoothstep(-1.0, 1.0, dist_bg);

            let body_d = capsule_sdf(px, py, center, body_top, center, body_bottom, body_radius);
            let stand_d = {
                let (dx, dy) = (px - center, py - stand_center_y);
                let ring = (dx * dx + dy * dy).sqrt() - stand_radius;
                // Only the lower half renders, leaving the ring open at the top.
                if dy >= 0.0 {
                    ring.abs() - stand_thickness / 2.0
                } else {
                    f32::MAX
                }
            };
            let leg_d = capsule_sdf(px, py, center, leg_top, center, leg_bottom, leg_radius);
            let base_d = capsule_sdf(
                px,
                py,
                center - base_half_width,
                leg_bottom,
                center + base_half_width,
                leg_bottom,
                base_radius,
            );

            let glyph_d = body_d.min(stand_d).min(leg_d).min(base_d);
            let alpha_glyph = 1.0 - smoothstep(-1.0, 1.0, glyph_d);

            let lerp = |a: f32, b: f32| a + (b - a) * alpha_glyph;
            rgba.push(lerp(BG.0, FG.0).round() as u8);
            rgba.push(lerp(BG.1, FG.1).round() as u8);
            rgba.push(lerp(BG.2, FG.2).round() as u8);
            rgba.push((alpha_bg * 255.0).round() as u8);
        }
    }
    Icon::from_rgba(rgba, SIZE as u32, SIZE as u32).expect("valid icon rgba")
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
