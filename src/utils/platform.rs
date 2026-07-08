//! Platform detection for the paste simulation.

fn is_linux() -> bool {
    cfg!(target_os = "linux")
}

fn is_wayland() -> bool {
    is_linux() && std::env::var("XDG_SESSION_TYPE").as_deref() == Ok("wayland")
}

/// Reports human-readable warnings about missing system dependencies needed to
/// simulate pasting on the current platform. Never fails: it only informs.
pub fn check_paste_dependencies() -> Vec<String> {
    let mut warnings = Vec::new();
    if is_wayland() {
        warnings.push(
            "Linux Wayland detected: keyboard simulation (Ctrl+V) may require extra \
             configuration (e.g. ydotool)."
                .to_string(),
        );
    }
    warnings
}
