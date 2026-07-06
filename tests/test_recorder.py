import asyncio

import numpy as np
import pytest

from config import VoiceCodeConfig
from domain.models import AudioChunk, KeyEvent
from pipeline.recorder import SoundDeviceRecorder


class FakeStream:
    def __init__(self, **kwargs: object) -> None:
        self.callback = kwargs['callback']
        self.samplerate = kwargs['samplerate']
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


def make_stream_factory() -> tuple[list[FakeStream], object]:
    created: list[FakeStream] = []

    def factory(**kwargs: object) -> FakeStream:
        stream = FakeStream(**kwargs)
        created.append(stream)
        return stream

    return created, factory


def feed_audio(stream: FakeStream, seconds: float) -> None:
    frames = int(stream.samplerate * seconds)
    chunk = np.zeros((frames, 1), dtype=np.float32)
    stream.callback(chunk, frames, None, None)


@pytest.mark.asyncio
async def test_seq_increments_per_recording() -> None:
    config = VoiceCodeConfig()
    created, factory = make_stream_factory()
    recorder = SoundDeviceRecorder(config, stream_factory=factory)
    key_queue: asyncio.Queue[KeyEvent] = asyncio.Queue()
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

    task = asyncio.create_task(recorder.record(key_queue, audio_queue))

    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await asyncio.sleep(0.01)
    feed_audio(created[0], seconds=1.0)
    await key_queue.put(KeyEvent(kind='up', key='f12'))
    first = await asyncio.wait_for(audio_queue.get(), timeout=1)

    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await asyncio.sleep(0.01)
    feed_audio(created[1], seconds=1.0)
    await key_queue.put(KeyEvent(kind='up', key='f12'))
    second = await asyncio.wait_for(audio_queue.get(), timeout=1)

    task.cancel()

    assert first.seq == 0
    assert second.seq == 1


@pytest.mark.asyncio
async def test_buffer_clears_on_new_recording_start() -> None:
    config = VoiceCodeConfig()
    created, factory = make_stream_factory()
    recorder = SoundDeviceRecorder(config, stream_factory=factory)
    key_queue: asyncio.Queue[KeyEvent] = asyncio.Queue()
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

    task = asyncio.create_task(recorder.record(key_queue, audio_queue))

    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await asyncio.sleep(0.01)
    feed_audio(created[0], seconds=1.0)
    await key_queue.put(KeyEvent(kind='up', key='f12'))
    first = await asyncio.wait_for(audio_queue.get(), timeout=1)

    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await asyncio.sleep(0.01)
    feed_audio(created[1], seconds=0.5)
    await key_queue.put(KeyEvent(kind='up', key='f12'))
    second = await asyncio.wait_for(audio_queue.get(), timeout=1)

    task.cancel()

    assert len(first.data) == config.sample_rate
    assert len(second.data) == config.sample_rate // 2


@pytest.mark.asyncio
async def test_discards_recording_shorter_than_min_duration() -> None:
    config = VoiceCodeConfig(min_audio_duration_ms=300)
    created, factory = make_stream_factory()
    recorder = SoundDeviceRecorder(config, stream_factory=factory)
    key_queue: asyncio.Queue[KeyEvent] = asyncio.Queue()
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

    task = asyncio.create_task(recorder.record(key_queue, audio_queue))

    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await asyncio.sleep(0.01)
    feed_audio(created[0], seconds=0.1)
    await key_queue.put(KeyEvent(kind='up', key='f12'))

    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await asyncio.sleep(0.01)
    feed_audio(created[1], seconds=1.0)
    await key_queue.put(KeyEvent(kind='up', key='f12'))
    only_item = await asyncio.wait_for(audio_queue.get(), timeout=1)

    task.cancel()

    assert audio_queue.empty()
    assert only_item.seq == 0


@pytest.mark.asyncio
async def test_ignores_duplicate_key_down_from_key_repeat() -> None:
    config = VoiceCodeConfig()
    created, factory = make_stream_factory()
    recorder = SoundDeviceRecorder(config, stream_factory=factory)
    key_queue: asyncio.Queue[KeyEvent] = asyncio.Queue()
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

    task = asyncio.create_task(recorder.record(key_queue, audio_queue))

    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await asyncio.sleep(0.01)
    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await asyncio.sleep(0.01)
    feed_audio(created[0], seconds=1.0)
    await key_queue.put(KeyEvent(kind='up', key='f12'))
    item = await asyncio.wait_for(audio_queue.get(), timeout=1)

    task.cancel()

    assert len(created) == 1
    assert created[0].closed is True
    assert item.seq == 0


@pytest.mark.asyncio
async def test_ignores_stray_key_up_without_prior_down() -> None:
    config = VoiceCodeConfig()
    created, factory = make_stream_factory()
    recorder = SoundDeviceRecorder(config, stream_factory=factory)
    key_queue: asyncio.Queue[KeyEvent] = asyncio.Queue()
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

    task = asyncio.create_task(recorder.record(key_queue, audio_queue))

    await key_queue.put(KeyEvent(kind='up', key='f12'))
    await asyncio.sleep(0.01)

    await key_queue.put(KeyEvent(kind='down', key='f12'))
    await asyncio.sleep(0.01)
    feed_audio(created[0], seconds=1.0)
    await key_queue.put(KeyEvent(kind='up', key='f12'))
    item = await asyncio.wait_for(audio_queue.get(), timeout=1)

    task.cancel()

    assert audio_queue.empty()
    assert item.seq == 0
    assert len(created) == 1
