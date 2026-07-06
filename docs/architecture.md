# Arquitectura

## Modelo de concurrencia

El sistema usa `asyncio` como event loop principal. Cada etapa del pipeline es una coroutine independiente que consume de una `asyncio.Queue` y produce en la siguiente. Los componentes bloqueantes (`pynput`, `sounddevice`, `faster-whisper`) corren en threads separados y se comunican con el event loop mediante mecanismos seguros para threads:

| Componente | Mecanismo thread → asyncio |
|---|---|
| `pynput` callbacks (Listener) | `loop.call_soon_threadsafe(queue.put_nowait, event)` |
| `sounddevice` callback (Recorder) | Lista de `numpy.ndarray` acumulada directamente en el callback (protegida por el GIL); el coroutine de `asyncio` la lee recién después de `stream.stop()` |
| `faster-whisper` (Transcriber) | `await loop.run_in_executor(executor, transcribe_fn, audio)` |

## Pipeline de datos

```
Listener → Recorder → Transcriber → Cleaner → Writer
```

Cada grabación genera un ítem que viaja por el pipeline portando su número de secuencia (`seq`) desde el origen hasta el writer. El `seq` nunca es modificado por etapas intermedias.

| Etapa | Entrada | Salida | Mecanismo |
|---|---|---|---|
| `PynputListener` | Evento OS (tecla) | `KeyEvent` en `key_queue` | Thread daemon de `pynput` + `call_soon_threadsafe` |
| `SoundDeviceRecorder` | `KeyEvent` de `key_queue` | `AudioChunk` en `audio_queue` | Thread de `sounddevice` + contador de `seq` |
| `WhisperTranscriber` | `AudioChunk` de `audio_queue` | `TranscribedText` en `text_queue` | `run_in_executor` + `ThreadPoolExecutor` |
| `RegexCleaner` | `TranscribedText` de `text_queue` | `CleanText` en `clean_queue` | Coroutine pura (sin I/O) |
| `ClipboardWriter` | `CleanText` de `clean_queue` | Texto en terminal activa | Coroutine + `pyperclip` + `pynput` |

Los `Protocol` de cada etapa están en `domain/protocols.py` (`KeyListenerProtocol`, `AudioRecorderProtocol`, `TranscriberProtocol`, `CleanerProtocol`, `WriterProtocol`) — permiten mockear cualquier etapa en tests sin modificar código de producción.

## Modelos de dominio

Todos son `dataclass(frozen=True)` en `domain/models.py`:

- `KeyEvent(kind: Literal['down', 'up'], key: str)`
- `AudioChunk(seq: int, data: np.ndarray, sample_rate: int)`
- `TranscribedText(seq: int, raw: str)`
- `CleanText(seq: int, text: str)`

## Secuenciamiento y orden de salida

El pipeline admite grabaciones solapadas: mientras la grabación #1 se transcribe, el usuario puede iniciar la grabación #2. Como `WhisperTranscriber.transcribe()` lanza cada transcripción como una `asyncio.Task` independiente (no es un worker secuencial), las transcripciones pueden completarse fuera de orden. El `ClipboardWriter` corrige esto con un **sequence buffer** interno (`pipeline/writer.py:SequenceBuffer`):

- Mantiene `expected_seq: int = 0` y `pending: dict[int, CleanText]`
- Al recibir un item con `seq == expected_seq` → lo agrega a la lista de listos y avanza `expected_seq`
- Tras avanzar, drena `pending` en orden ascendente mientras haya items contiguos
- Al recibir un item con `seq > expected_seq` → lo guarda en `pending[seq]`
- Invariante: nunca llega un `seq < expected_seq` (el `Recorder` es la única fuente de `seq`, y solo incrementa)

`SequenceBuffer` está deliberadamente separado de la lógica de clipboard para poder testearlo puro, sin mocks de `pyperclip`/`pynput` (ver [testing.md](testing.md)).

### Por qué el Recorder no puede simplemente ignorar eventos de tecla

`SoundDeviceRecorder` mantiene un flag `_recording: bool` además del buffer. Sin él, dos escenarios rompen el pipeline:

- **Key-repeat del OS**: mantener la tecla PTT apretada puede disparar varios `on_press` antes del único `on_release` (comportamiento normal de `pynput`/Windows). Sin el guard, cada `down` extra crearía un `InputStream` nuevo pisando la referencia al anterior, dejando streams de audio huérfanos.
- **Eventos sueltos**: un `up` sin `down` previo, sin el guard, igual concatenaría el buffer (con datos de la grabación anterior) y publicaría un `AudioChunk` duplicado.

`_start_recording()` ignora un `down` si ya está grabando; `_stop_recording()` ignora un `up` si no lo está.

## Manejo de errores

- **Listener**: loguea la excepción y continúa — nunca detiene el pipeline (una excepción en `on_press`/`on_release` no debe tumbar la escucha global de teclado).
- **Recorder**: descarta grabaciones más cortas que `min_audio_duration_ms` (grabaciones accidentales).
- **Transcriber**: si `faster-whisper` lanza una excepción, se loguea y se descarta ese chunk — el pipeline sigue procesando el resto. Si el resultado es texto vacío o solo espacios, también se descarta sin publicar.
- **Writer**: si el `CleanText` recibido tiene texto vacío (por ejemplo, un audio que era pura muletilla), se salta la operación de clipboard pero el `seq` igual se cuenta para no trabar el drenado del `pending`.
- **Writer (clipboard)**: el backup/restore del portapapeles usa `try/finally`, así que el contenido original se restaura incluso si algo falla entre el `copy` y el paste simulado.

## Composition root

`main.py` instancia cada componente con inyección de dependencias manual (sin frameworks de DI), crea las 4 colas del pipeline y lanza las 5 coroutines con `asyncio.gather`:

```python
config = load_config()

listener = PynputListener(config)
recorder = SoundDeviceRecorder(config)
transcriber = WhisperTranscriber(config)
cleaner = RegexCleaner(config)
writer = ClipboardWriter(config)

await asyncio.gather(
    listener.listen(key_queue),
    recorder.record(key_queue, audio_queue),
    transcriber.transcribe(audio_queue, text_queue),
    cleaner.clean(text_queue, clean_queue),
    writer.write(clean_queue),
)
```
