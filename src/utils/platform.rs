//! Detección de plataforma (== `utils/platform.py`).
//!
//! Nota: el hack `add_nvidia_dll_directories()` del proyecto Python es
//! específico de CTranslate2 y **no se porta**. Con el backend local
//! (whisper-rs) las DLLs del runtime CUDA deben ser visibles vía `PATH` o estar
//! junto al ejecutable (ver docs de empaquetado).

pub fn is_windows() -> bool {
    cfg!(target_os = "windows")
}

pub fn is_linux() -> bool {
    cfg!(target_os = "linux")
}

pub fn is_wayland() -> bool {
    is_linux() && std::env::var("XDG_SESSION_TYPE").as_deref() == Ok("wayland")
}

/// Devuelve advertencias legibles sobre dependencias de sistema faltantes para
/// simular el pegado en la plataforma actual. No falla: solo informa.
pub fn check_paste_dependencies() -> Vec<String> {
    let mut warnings = Vec::new();
    if is_wayland() {
        warnings.push(
            "Linux Wayland detectado: la simulación de teclado (Ctrl+V) puede requerir \
             configuración adicional (equivalente a ydotool)."
                .to_string(),
        );
    }
    warnings
}
