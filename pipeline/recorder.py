import asyncio
import logging
from typing import Any, Callable

import numpy as np
import sounddevice as sd

from config import VoiceCodeConfig
from domain.models import AudioChunk, KeyEvent
from utils.audio import concatenate_chunks, duration_ms

logger = logging.getLogger(__name__)

StreamFactory = Callable[..., Any]


class SoundDeviceRecorder:
    def __init__(
        self,
        config: VoiceCodeConfig,
        stream_factory: StreamFactory = sd.InputStream,
    ) -> None:
        self._config = config
        self._stream_factory = stream_factory
        self._seq = 0
        self._buffer: list[np.ndarray] = []
        self._stream: Any = None
        self._recording = False

    async def record(
        self,
        key_queue: asyncio.Queue[KeyEvent],
        audio_queue: asyncio.Queue[AudioChunk],
    ) -> None:
        while True:
            event = await key_queue.get()
            if event.kind == 'down':
                self._start_recording()
            elif event.kind == 'up':
                await self._stop_recording(audio_queue)
            key_queue.task_done()

    def _start_recording(self) -> None:
        if self._recording:
            logger.debug('Ignoring duplicate key-down while already recording (key repeat)')
            return
        self._recording = True
        self._buffer = []
        self._stream = self._stream_factory(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype='float32',
            callback=self._audio_callback,
        )
        self._stream.start()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        if status:
            logger.warning('Audio input status: %s', status)
        self._buffer.append(indata[:, 0].copy())

    async def _stop_recording(self, audio_queue: asyncio.Queue[AudioChunk]) -> None:
        if not self._recording:
            logger.debug('Ignoring key-up while not recording')
            return
        self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        samples = concatenate_chunks(self._buffer)
        if duration_ms(samples, self._config.sample_rate) < self._config.min_audio_duration_ms:
            logger.info('Discarding recording shorter than min_audio_duration_ms')
            return

        seq = self._seq
        self._seq += 1
        await audio_queue.put(
            AudioChunk(seq=seq, data=samples, sample_rate=self._config.sample_rate)
        )
