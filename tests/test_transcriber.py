import asyncio
import time

import numpy as np
import pytest

from config import VoiceCodeConfig
from domain.models import AudioChunk, TranscribedText
from pipeline.transcriber import WhisperTranscriber


class CountingWhisperModel:
    """Factory-friendly model that records how many times it is instantiated."""

    instances = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        type(self).instances += 1

    def transcribe(self, audio: np.ndarray, language: str) -> tuple[list['FakeSegment'], None]:
        return [FakeSegment('hola mundo')], None


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisperModel:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def transcribe(self, audio: np.ndarray, language: str) -> tuple[list[FakeSegment], None]:
        return [FakeSegment('hola mundo')], None


class EmptyWhisperModel:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def transcribe(self, audio: np.ndarray, language: str) -> tuple[list[FakeSegment], None]:
        return [FakeSegment('   ')], None


class RaisingWhisperModel:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def transcribe(self, audio: np.ndarray, language: str) -> tuple[list[FakeSegment], None]:
        raise RuntimeError('boom')


def make_chunk(seq: int) -> AudioChunk:
    return AudioChunk(seq=seq, data=np.zeros(1600, dtype=np.float32), sample_rate=16000)


@pytest.mark.asyncio
async def test_publishes_transcribed_text_for_valid_audio() -> None:
    config = VoiceCodeConfig()
    transcriber = WhisperTranscriber(config, model_factory=FakeWhisperModel)
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
    text_queue: asyncio.Queue[TranscribedText] = asyncio.Queue()

    await audio_queue.put(make_chunk(0))
    task = asyncio.create_task(transcriber.transcribe(audio_queue, text_queue))

    result = await asyncio.wait_for(text_queue.get(), timeout=1)
    task.cancel()

    assert result == TranscribedText(seq=0, raw='hola mundo')


@pytest.mark.asyncio
async def test_discards_empty_transcription() -> None:
    config = VoiceCodeConfig()
    transcriber = WhisperTranscriber(config, model_factory=EmptyWhisperModel)
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
    text_queue: asyncio.Queue[TranscribedText] = asyncio.Queue()

    await audio_queue.put(make_chunk(0))
    task = asyncio.create_task(transcriber.transcribe(audio_queue, text_queue))
    await asyncio.sleep(0.05)
    task.cancel()

    assert text_queue.empty()


@pytest.mark.asyncio
async def test_swallows_exception_and_keeps_running() -> None:
    config = VoiceCodeConfig()
    transcriber = WhisperTranscriber(config, model_factory=RaisingWhisperModel)
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
    text_queue: asyncio.Queue[TranscribedText] = asyncio.Queue()

    await audio_queue.put(make_chunk(0))
    task = asyncio.create_task(transcriber.transcribe(audio_queue, text_queue))
    await asyncio.sleep(0.05)

    assert not task.done()
    assert text_queue.empty()

    task.cancel()


@pytest.mark.asyncio
async def test_overlapping_transcriptions_all_complete_out_of_order_or_not() -> None:
    config = VoiceCodeConfig(transcriber_max_workers=2)
    transcriber = WhisperTranscriber(config, model_factory=FakeWhisperModel)
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
    text_queue: asyncio.Queue[TranscribedText] = asyncio.Queue()

    for seq in range(3):
        await audio_queue.put(make_chunk(seq))

    task = asyncio.create_task(transcriber.transcribe(audio_queue, text_queue))

    results = [await asyncio.wait_for(text_queue.get(), timeout=1) for _ in range(3)]
    task.cancel()

    assert {r.seq for r in results} == {0, 1, 2}


@pytest.mark.asyncio
async def test_tracks_inflight_tasks_to_prevent_premature_gc() -> None:
    config = VoiceCodeConfig()
    transcriber = WhisperTranscriber(config, model_factory=FakeWhisperModel)
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
    text_queue: asyncio.Queue[TranscribedText] = asyncio.Queue()

    await audio_queue.put(make_chunk(0))
    task = asyncio.create_task(transcriber.transcribe(audio_queue, text_queue))

    await asyncio.wait_for(text_queue.get(), timeout=1)
    await asyncio.sleep(0.01)

    assert len(transcriber._tasks) == 0

    task.cancel()


@pytest.mark.asyncio
async def test_model_is_loaded_lazily_on_first_transcription() -> None:
    CountingWhisperModel.instances = 0
    config = VoiceCodeConfig()
    transcriber = WhisperTranscriber(config, model_factory=CountingWhisperModel)

    # Nada se carga hasta que llega el primer audio.
    assert CountingWhisperModel.instances == 0

    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
    text_queue: asyncio.Queue[TranscribedText] = asyncio.Queue()
    await audio_queue.put(make_chunk(0))
    task = asyncio.create_task(transcriber.transcribe(audio_queue, text_queue))

    await asyncio.wait_for(text_queue.get(), timeout=1)
    task.cancel()

    assert CountingWhisperModel.instances == 1


def test_maybe_unload_frees_idle_model_and_reloads_on_demand() -> None:
    CountingWhisperModel.instances = 0
    config = VoiceCodeConfig(idle_unload_seconds=300)
    transcriber = WhisperTranscriber(config, model_factory=CountingWhisperModel)

    # Primer uso carga el modelo.
    transcriber._run_model(np.zeros(1600, dtype=np.float32))
    assert transcriber._model is not None
    assert CountingWhisperModel.instances == 1

    # Aún no ha pasado el tiempo de inactividad: no se descarga.
    assert transcriber._maybe_unload() is False
    assert transcriber._model is not None

    # Simulamos que el último uso fue hace más del umbral.
    transcriber._last_used = time.monotonic() - 301
    assert transcriber._maybe_unload() is True
    assert transcriber._model is None

    # El siguiente uso vuelve a cargar el modelo (segunda instancia).
    transcriber._run_model(np.zeros(1600, dtype=np.float32))
    assert transcriber._model is not None
    assert CountingWhisperModel.instances == 2


def test_maybe_unload_does_nothing_while_transcription_active() -> None:
    CountingWhisperModel.instances = 0
    config = VoiceCodeConfig(idle_unload_seconds=300)
    transcriber = WhisperTranscriber(config, model_factory=CountingWhisperModel)

    transcriber._acquire_model()  # marca una transcripción en curso (no la libera)
    transcriber._last_used = time.monotonic() - 301

    # Con _active > 0 no debe descargar aunque haya pasado el umbral.
    assert transcriber._maybe_unload() is False
    assert transcriber._model is not None
