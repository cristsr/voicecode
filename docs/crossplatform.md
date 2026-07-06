# Consideraciones Crossplatform

VoiceCode corre hoy en Windows. El diseño apunta a que la migración a Linux no requiera cambios de código Python — solo instalar dependencias de sistema. La abstracción de detección de OS vive en `utils/platform.py`.

## Soporte por componente

| Componente | Windows | Linux X11 | Linux Wayland |
|---|---|---|---|
| `pynput` (teclas globales) | ✅ Nativo | ✅ `python-xlib` | ✅ `pynput` ≥ 1.7 |
| `sounddevice` (audio) | ✅ PortAudio | ✅ PortAudio | ✅ PortAudio |
| `faster-whisper` CUDA | ✅ `pip install -e ".[gpu]"` | ✅ `pip install -e ".[gpu]"` | ✅ `pip install -e ".[gpu]"` |
| `pyperclip` (clipboard) | ✅ win32 | ✅ `xclip`/`xsel` | ✅ `wl-clipboard` |
| `pynput` (simular Ctrl+V) | ✅ Nativo | ✅ X11 | ⚠️ Requiere `ydotool` |

`pyperclip` detecta automáticamente el backend de clipboard disponible (win32 / xclip / xsel / wl-clipboard) — no hay lógica condicional en el código de VoiceCode para esto.

## Dependencias de sistema

- **Driver NVIDIA actualizado** + `pip install -e ".[gpu]"` (instala `nvidia-cublas-cu12` y `nvidia-cudnn-cu12` vía pip), en cualquier plataforma, para correr `faster-whisper` en GPU. No hace falta el CUDA Toolkit completo de NVIDIA — solo esas dos librerías de runtime.
- **Windows**: sin extras de sistema más allá de lo anterior.
- **Linux X11**: `python-xlib`, y `xclip` o `xsel` para el clipboard.
- **Linux Wayland**: `wl-clipboard` para el clipboard, y **`ydotool`** para simular el `Ctrl+V` (la simulación de teclado nativa de `pynput` no funciona en Wayland por diseño del protocolo — es la única pieza que realmente requiere una herramienta externa).

## `utils/platform.py`

Provee:

- `is_windows()` / `is_linux()` / `is_wayland()` — detección de plataforma vía `platform.system()` y la variable de entorno `XDG_SESSION_TYPE`.
- `check_paste_dependencies() -> list[str]` — devuelve advertencias legibles si falta una dependencia de sistema necesaria para simular el paste en la plataforma actual (hoy solo chequea `ydotool` en Wayland vía `shutil.which`). No lanza excepción — solo informa; es responsabilidad de quien llame decidir qué hacer con la advertencia (por ejemplo, loguearla al arrancar `main.py`).
- `add_nvidia_dll_directories() -> list[str]` — **Windows únicamente**. `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` instalados por pip dejan sus DLLs dentro de `site-packages/nvidia/*/bin/`, una ruta que Windows no busca al cargar bibliotecas dinámicas (a diferencia de Linux, que sí resuelve rutas dentro de site-packages). Sin este fix, `ctranslate2` falla en tiempo de inferencia con `Library cublas64_12.dll is not found`, aunque el paquete esté instalado y el modelo haya cargado bien. La función registra esos directorios con `os.add_dll_directory()` y además los antepone a `PATH` — se necesitan ambas cosas porque la carga de DLLs de `ctranslate2` no respeta `AddDllDirectory` de forma consistente, pero sí respeta `PATH` siempre. Se llama automáticamente desde `pipeline/transcriber.py` antes de importar `faster_whisper`, así que no requiere ninguna acción manual.

## Qué falta para migrar a Linux

1. Instalar las dependencias de sistema de la tabla de arriba según X11 o Wayland.
2. En Wayland, instalar `ydotool` y verificar que el daemon (`ydotoold`) esté corriendo — sin esto, el paso de "simular Ctrl+V" del `Writer` no tiene efecto aunque el resto del pipeline funcione.
3. No se requiere ningún cambio en `pipeline/`, `domain/` ni `config.py` — toda la diferencia es de entorno, no de código.
