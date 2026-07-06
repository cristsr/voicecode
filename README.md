# VoiceCode

Voice-to-Code Pipeline — controla Claude Code y OpenCode por voz usando Push-to-Talk (PTT). Mantenés presionada una tecla, hablás, la soltás, y el texto transcrito y limpio de muletillas se pega directamente en la terminal activa.

Python 3.11+ • asyncio • faster-whisper • CUDA

## Características

- Push-to-Talk con tecla configurable (`F12` por defecto)
- Transcripción local con `faster-whisper large-v3` en GPU (CUDA)
- Filtrado de muletillas por lista de expresiones regulares, configurable
- Pipeline asíncrono con solapamiento real de grabaciones concurrentes
- Orden de salida garantizado por número de secuencia, incluso con transcripciones que terminan fuera de orden
- Entrega vía clipboard con backup/restore (no pisa el contenido previo del portapapeles)
- Crossplatform: Windows (soportado) y Linux X11/Wayland (ver [docs/crossplatform.md](docs/crossplatform.md))
- Sin frameworks de DI externos: Python puro con `Protocol` para inversión de dependencias
- Empaquetable como app de fondo en Windows con ícono de bandeja, sin consola (ver [docs/packaging.md](docs/packaging.md))

## Requisitos

- Python 3.11+
- GPU NVIDIA con driver actualizado (no hace falta instalar el CUDA Toolkit completo — ver [Configuración de GPU](#configuración-de-gpu) abajo)
- Windows: sin dependencias extra de sistema
- Linux: ver [docs/crossplatform.md](docs/crossplatform.md) (`python-xlib`, `xclip`/`xsel` o `wl-clipboard` + `ydotool` según X11/Wayland)

## Instalación

```bash
pip install -e .
# o, para desarrollo (incluye pytest, pytest-asyncio, pytest-mock)
pip install -e ".[dev]"
```

## Configuración de GPU

`faster-whisper` (vía `ctranslate2`) necesita las librerías de runtime **cuBLAS** y **cuDNN** — no el CUDA Toolkit completo de NVIDIA (con `nvcc`, samples, etc., que acá no se usa para nada). Alcanza con:

```bash
pip install -e ".[gpu]"
```

Eso instala `nvidia-cublas-cu12` y `nvidia-cudnn-cu12` (paquetes pip, sin instalador aparte). El driver de la GPU es retrocompatible: si tenés un driver que reporta soporte CUDA 13.x (`nvidia-smi`), igual corre sin problema estas librerías de CUDA 12, que es contra lo que está compilado `ctranslate2` hoy.

En Windows, estas librerías quedan instaladas dentro de `site-packages/nvidia/*/bin/`, una ubicación que Windows no busca automáticamente al cargar DLLs (a diferencia de Linux). `pipeline/transcriber.py` resuelve esto solo, llamando a `utils.platform.add_nvidia_dll_directories()` antes de cargar el modelo — no hace falta editar el `PATH` del sistema a mano.

El modelo de Whisper en sí (`large-v3` por defecto, ~3GB) se descarga aparte, automáticamente, la primera vez que se instancia `WhisperTranscriber` — no es parte de la instalación de pip.

## Uso

```bash
python main.py
```

Mantené presionada la tecla configurada (`F12` por defecto), hablá, y soltá. El texto transcrito y limpio se pega automáticamente en la ventana/terminal activa.

### Como app de fondo, sin consola (con ícono de bandeja)

```bash
pip install -e ".[tray]"
python tray_app.py
```

Corre el mismo pipeline en un thread separado, con un ícono en la bandeja del sistema (*Reiniciar pipeline* / *Salir*) en vez de una terminal. Para empaquetarlo como `.exe` standalone y que arranque solo al iniciar sesión de Windows (sin necesidad de un servicio de Windows, que no puede acceder al teclado/clipboard de la sesión interactiva), ver [docs/packaging.md](docs/packaging.md).

## Configuración

Toda la configuración vive en [`config.toml`](config.toml), en la raíz del proyecto — se edita directamente, no hace falta tocar código ni reiniciar más que el proceso de VoiceCode. Ver el detalle completo de cada campo en [docs/configuration.md](docs/configuration.md).

Si `config.toml` no existe o le faltan secciones, `load_config()` completa con los defaults de `VoiceCodeConfig` — no rompe el arranque.

## Arquitectura

El pipeline tiene 5 etapas conectadas por `asyncio.Queue`, cada una una coroutine independiente:

```
Listener → Recorder → Transcriber → Cleaner → Writer
```

Ver el detalle completo (modelo de concurrencia, contratos de cada módulo, algoritmo del sequence buffer) en [docs/architecture.md](docs/architecture.md).

## Testing

```bash
pip install -e ".[dev,tray]"
python -m pytest tests/ -v
```

44 tests, todos ejecutables sin GPU (el módulo `transcriber.py` usa inyección de dependencias para poder testear la lógica de orquestación con un modelo Whisper falso, sin necesitar CUDA real). El extra `tray` es necesario para correr `tests/test_tray_app.py` (importa `pystray`/`Pillow`). Ver estrategia completa en [docs/testing.md](docs/testing.md).

## Estructura del proyecto

```
voicecode/
├── main.py                    # Composition root: run_pipeline() + entry point CLI
├── tray_app.py                 # Entry point de bandeja (sin consola) para empaquetar
├── config.py                  # VoiceCodeConfig + load_config() desde TOML
├── config.toml                # Configuración editable en runtime
├── domain/
│   ├── models.py               # AudioChunk, TranscribedText, CleanText, KeyEvent
│   └── protocols.py             # Protocol de cada componente del pipeline
├── pipeline/
│   ├── listener.py              # PTT key listener (pynput)
│   ├── recorder.py              # Audio recorder (sounddevice)
│   ├── transcriber.py           # faster-whisper transcriber
│   ├── cleaner.py               # Filtrado de muletillas
│   └── writer.py                # Clipboard writer con sequence buffer
├── utils/
│   ├── audio.py                 # Helpers de buffer y formato de audio
│   └── platform.py              # Abstracción crossplatform (Win/Linux)
├── scripts/                    # build_exe.ps1, register_task.ps1, unregister_task.ps1
├── tests/                      # 44 tests, ver docs/testing.md
└── docs/
    ├── architecture.md
    ├── configuration.md
    ├── testing.md
    ├── crossplatform.md
    └── packaging.md
```

## Limitaciones conocidas

- `transcriber.py` requiere GPU NVIDIA + CUDA real para funcionar en producción — no hay fallback a CPU probado en este repo (aunque `whisper_device = "cpu"` es configurable en `config.toml` para pruebas).
- No hay hot-reload de `config.toml` mientras el proceso corre — los cambios toman efecto en el próximo arranque.
- Corriendo con `python main.py`, no hay shutdown ordenado por señal (`Ctrl+C` corta el proceso sin cerrar streams de audio explícitamente). `tray_app.py` sí cancela la task del pipeline de forma limpia al usar *Salir*/*Reiniciar*.
- Empaquetado como tarea de Task Scheduler, no hay auto-restart si el pipeline muere por una excepción no controlada a mitad de sesión — ver la sección de limitaciones en [docs/packaging.md](docs/packaging.md).
