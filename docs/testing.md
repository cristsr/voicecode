# Testing

## Cómo correr los tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

39 tests, todos corren sin GPU y sin hardware de audio real — cada módulo con dependencias externas bloqueantes (`sounddevice`, `faster-whisper`, `pyperclip`/`pynput`) se testea con una dependencia inyectada en lugar de la real.

## Estrategia por módulo

| Módulo | Archivo de test | Tipo | Qué se verifica |
|---|---|---|---|
| `pipeline/cleaner.py` | `test_cleaner.py` (16 tests) | Unitario puro | Cada patrón de muletilla se remueve correctamente, texto vacío no rompe, puntuación colgante se limpia, capitalización |
| `config.py` | `test_config.py` (5 tests) | Unitario puro | Archivo ausente → defaults, TOML completo sobreescribe todo, TOML parcial mezcla con defaults, secciones faltantes se ignoran |
| `pipeline/writer.py` (buffer) | `test_writer_sequence.py` (5 tests) | Unitario puro | Orden garantizado con llegada desordenada, drenado de `pending` correcto, se detiene en el primer hueco |
| `pipeline/writer.py` (clipboard) | `test_writer_clipboard.py` (2 tests) | Unitario con mock | `_emit` se saltea para texto vacío pero el `seq` igual avanza |
| `pipeline/recorder.py` | `test_recorder.py` (5 tests) | Unitario con mock | `seq` incrementa por grabación, buffer se limpia al iniciar, descarta grabaciones cortas, ignora `down` duplicado (key-repeat) y `up` sin `down` previo |
| `pipeline/transcriber.py` | `test_transcriber.py` (5 tests) | Unitario con mock | Publica texto transcrito, descarta transcripción vacía, no se cae ante una excepción del modelo, solapamiento real de transcripciones concurrentes, las tasks in-flight se trackean (no quedan huérfanas para el GC) |
| Pipeline completo | `test_pipeline_e2e.py` (1 test) | Integración E2E | `AudioChunk` sintéticos → texto emitido en el orden correcto, usando el `mock_transcriber` de `conftest.py` |

`pipeline/listener.py` no tiene test dedicado — su única lógica no trivial es el bridge `pynput` → `asyncio` vía `call_soon_threadsafe`, que requiere un listener de teclado real para probarse con sentido; queda fuera del alcance de tests automatizados, igual que la transcripción real con GPU.

## Patrones de mock / DI usados

### Inyección de fábricas en vez de mocks pesados

`SoundDeviceRecorder` y `WhisperTranscriber` no reciben un mock de la librería completa — reciben una **fábrica inyectable** con el mismo shape que la dependencia real:

```python
# recorder.py
class SoundDeviceRecorder:
    def __init__(self, config, stream_factory: StreamFactory = sd.InputStream) -> None: ...
```

```python
# transcriber.py
class WhisperTranscriber:
    def __init__(self, config, model_factory: ModelFactory | None = None) -> None:
        ...
        if model_factory is None:
            from faster_whisper import WhisperModel
            model_factory = WhisperModel
```

El import de `faster_whisper` es perezoso (dentro del `if`, no a nivel de módulo) — así `pipeline/transcriber.py` se puede importar y testear sin tener `faster-whisper` instalado, siempre que los tests pasen su propio `model_factory` falso. Ver `tests/test_recorder.py::FakeStream` y `tests/test_transcriber.py::FakeWhisperModel` para los fakes usados.

### `SequenceBuffer` separado de `ClipboardWriter`

`pipeline/writer.py` separa la lógica pura de reordenamiento (`SequenceBuffer.process()`) de la I/O real de clipboard (`ClipboardWriter._emit()`). Esto permite testear el algoritmo de reordenamiento (la parte con más casos borde: huecos, drenado, orden) sin nunca tocar `pyperclip` ni `pynput` en el 90% de los tests — solo `test_writer_clipboard.py` mockea `_emit` directamente (con `unittest.mock.AsyncMock`) para verificar que se saltea con texto vacío.

### Fixture principal (`conftest.py`)

```python
@pytest.fixture
def mock_transcriber() -> TranscriberProtocol:
    class FakeTranscriber:
        async def transcribe(self, audio_q, text_q):
            chunk = await audio_q.get()
            await text_q.put(TranscribedText(seq=chunk.seq, raw='texto de prueba'))
    return FakeTranscriber()
```

`conftest.py` también provee fixtures de las 4 colas tipadas (`key_queue`, `audio_queue`, `text_queue`, `clean_queue`) para quien quiera componer un test de integración parcial sin repetir el boilerplate de `asyncio.Queue()`.
