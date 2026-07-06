# Empaquetado como app de fondo en Windows

## Por qué no es un servicio de Windows real

Un servicio de Windows (registrado vía SCM, `nssm` o `pywin32`) corre en la **Sesión 0**, aislada del escritorio interactivo desde Windows Vista. Un proceso en Sesión 0 no puede:

- Recibir el hook de teclado global de `pynput` (no ve lo que el usuario tipea en su sesión).
- Leer o escribir el portapapeles de la sesión interactiva.
- Simular `Ctrl+V` sobre la ventana que el usuario tiene enfocada.

Como las tres cosas son el corazón de VoiceCode, empaquetarlo como servicio de Windows en sentido estricto simplemente no funciona — el proceso arrancaría, pero nunca vería una tecla ni pegaría nada.

La alternativa correcta para "corre solo, en segundo plano, sin que yo tenga que abrir una terminal" es una **tarea de Task Scheduler que arranca al iniciar sesión, en la sesión interactiva del usuario** (`LogonType Interactive`) — más un ícono de bandeja para poder ver que está corriendo y detenerlo sin Task Manager.

## Componentes

- **`tray_app.py`**: entry point con ícono en la bandeja del sistema (`pystray`). Corre el pipeline completo (`main.run_pipeline`) en un thread con su propio event loop de `asyncio`, separado del thread del ícono. Menú: *Reiniciar pipeline* y *Salir*.
- **`scripts/build_exe.ps1`**: empaqueta `tray_app.py` a `dist/VoiceCode/VoiceCode.exe` con PyInstaller (`--onedir --noconsole`, con `--collect-binaries` para las DLLs de cuBLAS/cuDNN), y copia `config.toml` al lado del `.exe` (a propósito, sin embeberlo en el bundle). No necesita privilegios de Administrador — pensado para correrse seguido mientras se itera.
- **`scripts/deploy.ps1`**: mueve el build ya generado de `dist/VoiceCode` a su ubicación final en `E:\Program Files\VoiceCode`. **Necesita PowerShell como Administrador** (ver más abajo). Separado de `build_exe.ps1` a propósito, para no requerir elevación en cada rebuild durante desarrollo.
- **`scripts/register_task.ps1`**: registra la tarea en Task Scheduler, apuntando por default a `E:\Program Files\VoiceCode\VoiceCode.exe`.
- **`scripts/unregister_task.ps1`**: la elimina.

## Paso a paso

### 1. Instalar dependencias de empaquetado

```powershell
pip install -e ".[tray]"
pip install pyinstaller
```

### 2. Construir el ejecutable

```powershell
./scripts/build_exe.ps1
```

Genera `dist/VoiceCode/VoiceCode.exe` (modo `--onedir`, no `--onefile`) y copia `config.toml` junto a él (`dist/VoiceCode/config.toml`). A partir de acá, **editás `dist/VoiceCode/config.toml`** para cambiar la tecla PTT, las muletillas, etc. — no hace falta reconstruir el `.exe` para eso (ver [configuration.md](configuration.md)).

> **Por qué `--onedir` y no `--onefile`**: un `.exe` de un solo archivo se descomprime en una carpeta temporal *distinta en cada arranque*, lo cual (a) suma latencia de inicio cada vez que Task Scheduler lo lanza, y (b) rompe la detección de las DLLs de cuBLAS/cuDNN, porque `add_nvidia_dll_directories()` necesita una ubicación fija junto al `.exe` para encontrarlas, no una carpeta temporal que cambia de nombre en cada corrida. `--onedir` deja todo (el `.exe`, las DLLs de NVIDIA, `config.toml`) en `dist/VoiceCode/`, de forma persistente.
>
> `config.py` detecta cuándo corre empaquetado (`sys.frozen`) y busca `config.toml` junto al `.exe` (`sys.executable`). `utils/platform.py` hace lo mismo para las DLLs de NVIDIA — por eso `build_exe.ps1` usa `--collect-binaries nvidia.cublas --collect-binaries nvidia.cudnn`, para que esas carpetas terminen empaquetadas ahí y no falte `cublas64_12.dll` en tiempo de ejecución (el mismo error que se corrigió para el modo desarrollo, pero que reaparece en un `.exe` empaquetado si no se las bundlea explícitamente).

### 3. Desplegar a Program Files (requiere Administrador)

```powershell
# Desde una PowerShell abierta como Administrador:
./scripts/deploy.ps1
```

Mueve `dist/VoiceCode` a `E:\Program Files\VoiceCode`. Esta carpeta tiene los mismos permisos restringidos que el `C:\Program Files` real del sistema (solo `Administrators`/`SYSTEM` pueden escribir ahí) — por eso este paso está separado de `build_exe.ps1` y necesita una terminal elevada. Si no la abriste como Administrador, el script falla temprano con un mensaje claro en vez de un `Access Denied` a mitad de copia.

### 4. Registrar la tarea de inicio de sesión

```powershell
./scripts/register_task.ps1
```

Por default apunta a `E:\Program Files\VoiceCode\VoiceCode.exe`. Para instalarlo en otra ruta:

```powershell
./scripts/register_task.ps1 -ExePath "C:\Tools\VoiceCode\VoiceCode.exe"
```

La tarea usa `LogonType Interactive` con el usuario actual — corre en tu sesión, no como `SYSTEM`, precisamente para poder usar teclado/clipboard.

Para arrancarlo ya, sin cerrar sesión:

```powershell
Start-ScheduledTask -TaskName "VoiceCode"
```

### 5. Verificar que está corriendo

Buscá el ícono azul en la bandeja del sistema (puede estar en los íconos ocultos, flechita `^` junto al reloj). Click derecho para *Reiniciar pipeline* o *Salir*.

Si algo falla, los logs quedan en `E:\Program Files\VoiceCode\voicecode.log` (con `--noconsole` no hay ventana de consola para ver `print`/logging, por eso `tray_app.py` redirige el logging a ese archivo cuando detecta que está empaquetado).

### 6. Desinstalar

```powershell
./scripts/unregister_task.ps1
```

Esto solo borra la tarea programada — si VoiceCode está corriendo en ese momento, hay que cerrarlo aparte desde el ícono de bandeja (*Salir*), o el script intenta detener la tarea primero con `Stop-ScheduledTask`.

## Limitaciones de este esquema

- **Sin auto-restart ante crash**: si el pipeline muere por una excepción no controlada dentro de `run_pipeline`, el ícono de bandeja queda vivo pero sin pipeline corriendo — hay que usar *Reiniciar pipeline* manualmente. Task Scheduler solo relanza el `.exe` al *iniciar sesión*, no si el proceso muere a mitad de la sesión.
- **Build + deploy por cambio de código**: cambios en `config.toml` (ya desplegado, en `E:\Program Files\VoiceCode\config.toml`) no requieren nada más, pero cambios en cualquier `.py` sí requieren `./scripts/build_exe.ps1` y después `./scripts/deploy.ps1` (como Administrador) para que lleguen al `.exe` que realmente corre.
- **`ExecutionTimeLimit` deshabilitado a propósito** en la tarea (`register_task.ps1` lo pone en `TimeSpan.Zero`) porque Task Scheduler por default mata tareas que corren más de 3 días — VoiceCode está pensado para correr indefinidamente durante la sesión.
