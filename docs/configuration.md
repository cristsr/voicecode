# Configuración

Toda la configuración vive en [`config.toml`](../config.toml), en la raíz del proyecto. Se carga al arrancar `main.py` vía `config.load_config()`.

Si el archivo no existe, o le faltan secciones o claves puntuales, `load_config()` completa con los defaults de `VoiceCodeConfig` (`config.py`) — nunca falla el arranque por configuración incompleta.

Los cambios en `config.toml` toman efecto en el **próximo arranque** del proceso — no hay hot-reload en caliente mientras VoiceCode está corriendo.

## `[ptt]`

| Clave | Tipo | Default | Descripción |
|---|---|---|---|
| `key` | string | `"f12"` | Tecla de Push-to-Talk. Debe ser un nombre válido de `pynput.keyboard.Key` (ej. `f1`..`f12`, `space`, `caps_lock`). |

## `[audio]`

| Clave | Tipo | Default | Descripción |
|---|---|---|---|
| `sample_rate` | int | `16000` | Frecuencia de muestreo en Hz. `faster-whisper` requiere 16000. |
| `channels` | int | `1` | Canales de audio. El `Recorder` solo toma el canal 0 aunque se configure más de 1 — ver [Limitaciones](#limitaciones). |
| `min_audio_duration_ms` | int | `300` | Duración mínima de grabación en milisegundos. Grabaciones más cortas se descartan (evita publicar toques accidentales de la tecla). |

## `[whisper]`

| Clave | Tipo | Default | Descripción |
|---|---|---|---|
| `model` | string | `"large-v3"` | Modelo de `faster-whisper` a cargar. Se carga una sola vez en el constructor de `WhisperTranscriber`, y se descarga solo (desde Hugging Face, ~3GB para `large-v3`) la primera vez que se instancia. |
| `device` | string | `"cuda"` | `"cuda"` para GPU NVIDIA (requiere `pip install -e ".[gpu]"`, ver [README](../README.md#configuración-de-gpu)), `"cpu"` para correr sin GPU (mucho más lento, útil para pruebas locales sin hardware). |
| `compute_type` | string | `"float16"` | Precisión de cómputo. `float16` para GPU; con `device = "cpu"` normalmente se usa `int8` o `float32`. |
| `language` | string | `"es"` | Idioma fijo — evita la detección automática de `faster-whisper`, reduciendo latencia. |

## `[transcriber]`

| Clave | Tipo | Default | Descripción |
|---|---|---|---|
| `max_workers` | int | `2` | Tamaño del `ThreadPoolExecutor` que ejecuta las transcripciones bloqueantes. Limita cuántas transcripciones corren en paralelo real, independientemente de cuántas grabaciones se hayan encolado. |

## `[cleaner]`

| Clave | Tipo | Default | Descripción |
|---|---|---|---|
| `filler_patterns` | list[string] | ver abajo | Lista de expresiones regulares (aplicadas case-insensitive) que se remueven del texto transcrito antes de emitirlo. |

Patrones por defecto:

```toml
filler_patterns = [
    "\\beh+\\b",
    "\\bmmm+\\b",
    "\\bo sea\\b",
    "\\bdigamos\\b",
    "\\bbásicamente\\b",
    "\\bpues\\b",
    "\\benton?ces\\b",
    "\\bla verdad\\b",
]
```

Para agregar una muletilla nueva, sumar un patrón a la lista (recordá escapar el `\` como `\\` en TOML). El post-procesamiento del `Cleaner` colapsa espacios múltiples, limpia puntuación colgante que quede al inicio (ej. una coma que sobrevive a remover la muletilla) y capitaliza la primera letra.

## `[writer]`

| Clave | Tipo | Default | Descripción |
|---|---|---|---|
| `clipboard_restore_delay_ms` | int | `50` | Espera, en milisegundos, entre simular el `Ctrl+V` y restaurar el contenido original del portapapeles. Debe ser suficiente para que la aplicación destino procese el paste antes del restore. |

## Limitaciones

- **`channels`**: aunque es configurable, `SoundDeviceRecorder._audio_callback` siempre toma `indata[:, 0]` (el primer canal). Configurar `channels = 2` no captura estéreo — descarta el segundo canal en silencio, sin warning.
- **Validación**: `load_config()` no valida que los valores tengan sentido (ej. `sample_rate` negativo, `key` que no exista en `pynput.keyboard.Key`) — un valor inválido falla recién al construir el componente que lo usa, no al cargar la configuración.
