# VoiceCode

Dictado por voz **Push-to-Talk** para controlar Claude Code, OpenCode y cualquier
app. Mantenés presionada una tecla, hablás, la soltás, y el texto transcrito y
limpio de muletillas se pega directamente en la ventana activa.

**Rust** · Tokio · whisper.cpp (GPU/CUDA) · RNNoise

> Reescritura completa en Rust de un prototipo previo en Python (faster-whisper);
> la versión Python fue retirada. Referencias residuales a "== `foo.py`" en los
> comentarios son notas de equivalencia histórica.

Arquitectura de pipeline por etapas desacopladas conectadas por canales
(`tokio::sync::mpsc`), modelos de dominio inmutables, dependencias externas detrás
de traits inyectables (para testear sin hardware), `SequenceBuffer` separado de la
I/O y manejo de errores por etapa:

```
Listener → Recorder → Transcriber → Cleaner → Writer
 (rdev)     (cpal)    (whisper/groq)  (regex)  (enigo+arboard)
```

## Características

- Push-to-Talk con tecla configurable (`F12` por defecto).
- Transcripción **local** con Whisper `large-v3` en GPU (CUDA), con fallback a CPU.
- **Supresión de ruido** de fondo (RNNoise) antes de transcribir.
- Precarga del modelo al arrancar: la primera dictada no espera la carga.
- Filtrado de muletillas configurable (regex).
- Pipeline concurrente con orden de salida garantizado por secuencia.
- Entrega vía portapapeles con backup/restore (no pisa lo que tenías copiado).
- Un único `.exe` (sin runtime ni intérprete), con ícono de bandeja y sin consola.

## Backends de transcripción

Seleccionables en `config.toml` (`[transcriber] backend`), detrás de features de
Cargo para no arrastrar toolchain que no uses:

| Backend | Feature | Requisitos de build | Notas |
|---|---|---|---|
| `groq` | `groq` (default) | ninguno nativo | API compatible-OpenAI. Necesita `GROQ_API_KEY`. Binario pequeño. |
| `local` | `local` | **LLVM 18 (libclang) + CMake + MSVC + CUDA Toolkit** | whisper.cpp vía `whisper-rs`, **GPU-first (CUDA)**: usa la GPU si hay y cae a CPU si no. Offline/privado. Carga perezosa + descarga por inactividad. |
| `local-cpu` | `local-cpu` | **LLVM 18 (libclang) + CMake + MSVC** | Igual que `local` pero **solo CPU** (sin CUDA). Para máquinas sin GPU NVIDIA. |

## Build

```bash
# Backend Groq (por defecto): sin toolchain nativo.
cargo build --release

# Backend local GPU-first (whisper.cpp + CUDA): requiere toolchain nativo + CUDA.
cargo build --release --features local

# Backend local solo-CPU (sin CUDA):
cargo build --release --features local-cpu
```

### Toolchain para el backend `local` (Windows, verificado 2026-07-07)

`whisper-rs-sys` compila whisper.cpp con CMake y genera bindings con `bindgen`
(libclang). En Windows/MSVC hay que darle el entorno correcto:

1. **MSVC + CMake**: instalados con Visual Studio 2022 (workload "Desktop
   development with C++"). CMake viene en
   `...\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`.
2. **LLVM 18** (aporta `libclang.dll`): `winget install LLVM.LLVM --version 18.1.8`.
   ⚠️ **Fijar la versión 18.** `bindgen 0.71` (el que trae `whisper-rs-sys 0.13`)
   **no es compatible con LLVM ≥ 20/22**: genera `whisper_full_params` como struct
   opaco y el build falla con `error[E0080] ... size_of ... - 264usize` (overflow).
   Con LLVM 18 los bindings se generan bien. *No* usar los bindings empaquetados
   (`WHISPER_DONT_GENERATE_BINDINGS=1`): son de Linux (glibc) y rompen en Windows.
3. **CUDA Toolkit** (solo para la feature `local`, GPU): `nvcc` en el PATH y
   `CUDA_PATH` apuntando a la instalación (p. ej. `...\CUDA\v13.3`).

Build en un shell con el entorno cargado (equivalente a `vcvars64.bat`):

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
set "PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
set "PATH=%PATH%;%CUDA_PATH%\bin"
rem CUDA 13 (CCCL) exige el preprocesador conforme de MSVC; sobrescribe el
rem CMAKE_CUDA_FLAGS por defecto de whisper-rs-sys (que trae un -fPIC inservible en MSVC).
set "CMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor"
cargo build --release --features local
```

Sin esa flag, el build CUDA falla con `fatal error C1189: ... /Zc:preprocessor`.
Para acelerar el build (y evitar errores de arquitecturas viejas que CUDA 13 ya no
soporta) se puede fijar solo la arquitectura de tu GPU, p. ej. RTX 3080 = sm_86:
`set "CMAKE_CUDA_ARCHITECTURES=86"`.

**GPU-first en runtime:** el backend carga con `use_gpu(true)`; whisper.cpp usa la
GPU si hay dispositivo CUDA y cae a CPU si no. El binario `local` depende de las
DLLs de runtime de CUDA (`cudart64_*.dll`, `cublas64_*.dll`, `cublasLt64_*.dll`);
deben estar en el PATH (el instalador del CUDA Toolkit agrega `...\CUDA\vX.Y\bin`)
o copiadas junto al `.exe`. El campo `[whisper] device` del `config.toml` no lo
usa el backend Rust (whisper.cpp autodetecta).

El ejecutable queda en `target/release/voicecode(.exe)`. Copiá `config.toml`
junto a él (se lee desde el directorio del ejecutable; si no está, usa defaults).

## Configuración

Copiá la plantilla y ajustala a tu equipo (el `config.toml` personal está
gitignoreado, no se versiona):

```bash
cp config.sample.toml config.toml
```

Se lee al arrancar el pipeline; **no hay hot-reload** (los cambios toman efecto al
reiniciar el proceso, o con *Reiniciar pipeline* del tray). Campos destacados:

- `[transcriber] backend = "groq" | "local"`
- `[audio] denoise` — supresión de ruido de fondo (RNNoise) antes de transcribir.
  Atenúa ventilador, teclado, hiss y ambiente conservando la voz. **No** separa
  hablantes: otras voces de fondo se conservan. Se aplica sobre el buffer grabado
  al soltar la tecla (a 48 kHz, la tasa de RNNoise), antes de bajar a 16 kHz.
- `[transcriber] idle_unload_seconds` — descarga del modelo local por inactividad.
- `[groq]` — `model`, `api_key_env` (la API key se lee de esa variable de entorno,
  nunca del archivo), `base_url`.
- `[whisper] model_path` — ruta al modelo GGML `.bin` (solo backend local).

## Correr

```bash
# Backend Groq
export GROQ_API_KEY=gsk_...        # PowerShell: $env:GROQ_API_KEY="gsk_..."
cargo run --release
```

Aparece un ícono en la bandeja del sistema (menú: *Reiniciar pipeline*, *Salir*).
Mantené la tecla PTT (`f12` por defecto) para dictar.

## Empaquetado y arranque automático (Windows)

`cargo build --release` produce un único `.exe`. Para que arranque solo al iniciar
sesión se reutilizan los scripts de Task Scheduler:

```powershell
# desde PowerShell como Administrador
.\scripts\register_task.ps1 -ExePath "C:\ruta\a\voicecode.exe"
```

### Sobre la elevación (UIPI)

Windows (UIPI) **solo** bloquea que un proceso escriba en una ventana de **mayor
integridad**. En la práctica:

- Si tu terminal (p. ej. Claude Code) corre **sin elevar**, VoiceCode **sin elevar
  pega sin problema** — no hace falta admin.
- Solo si necesitás pegar en una ventana **elevada** (una terminal abierta *como
  administrador*), VoiceCode también debe correr elevado. Para eso la tarea
  programada usa `RunLevel Highest` (arranca elevada al logon, sin prompt de UAC).
  Un acceso directo en `shell:startup` **no** sirve para esto: arranca con token
  estándar (no elevado).

Con el backend local, las DLLs del runtime CUDA deben ser visibles (en `PATH` o
junto al `.exe`).

## Testing

```bash
cargo test              # suite completa (sin GPU/audio/red, todo con fakes)
cargo clippy --all-targets
cargo fmt --check
```

Prueba de integración del backend local contra el modelo real (usa GPU si está
disponible). Se **salta sola** si no encontrás el modelo en
`models/ggml-large-v3.bin` (o `VOICECODE_TEST_MODEL`) y el sample `models/jfk.wav`:

```bat
rem con el entorno de build cargado (ver sección toolchain)
cargo test --release --features local --test local_gpu -- --nocapture
```

Los logs de whisper.cpp confirman la GPU: `use gpu = 1`,
`found 1 CUDA devices`, `using device CUDA0 (...)`.

## Estructura

```
voicecode/
├── Cargo.toml · Cargo.lock
├── config.toml              # configuración (tecla, backend, denoise, model_path...)
├── src/                     # crate `voicecode`
│   ├── main.rs              # entry point + ícono de bandeja
│   ├── lib.rs               # composition root (run_pipeline)
│   ├── config.rs · domain/ · utils/
│   └── pipeline/            # listener · recorder · transcriber/ · cleaner · writer
├── examples/                # groq_check.rs
├── tests/                   # e2e.rs · local_gpu.rs (integración GPU, se salta sin modelo)
├── scripts/                 # register_task.ps1 · unregister_task.ps1 (Task Scheduler)
└── models/                  # modelos GGML (.bin), no versionados
```

## Limitaciones conocidas

- **Reiniciar pipeline** desde el tray: `rdev` no permite detener la escucha
  global de teclado, así que el thread del listener anterior persiste. El uso
  normal (arrancar una vez al iniciar sesión) no se ve afectado; *Salir* cierra
  el proceso por completo.
- El backend `local` **compila, usa GPU y transcribe** (verificado 2026-07-07:
  large-v3 GGML cargado en una RTX 3080 vía CUDA, transcripción correcta del
  sample `jfk.wav` en ~3.75 s incluyendo la carga del modelo). Requiere LLVM 18
  (con ≥ 20 el build de bindings falla) y la flag `/Zc:preprocessor` para CUDA 13
  — ver la sección de toolchain.
- **Supresión de ruido**: RNNoise atenúa ruido pero **no separa hablantes**; otras
  voces de fondo se conservan.
