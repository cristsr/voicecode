import asyncio
import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from config import VoiceCodeConfig
from domain.models import AudioChunk, TranscribedText
from utils.platform import add_nvidia_dll_directories

logger = logging.getLogger(__name__)

ModelFactory = Callable[..., Any]

# Cadencia con la que el monitor revisa si el modelo lleva demasiado tiempo ocioso.
_IDLE_CHECK_INTERVAL_S = 30.0


class WhisperTranscriber:
    def __init__(
        self,
        config: VoiceCodeConfig,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=config.transcriber_max_workers)
        if model_factory is None:
            add_nvidia_dll_directories()
            from faster_whisper import WhisperModel
            model_factory = WhisperModel
        self._model_factory = model_factory
        # El modelo se carga de forma perezosa en la primera transcripción y se
        # libera de la GPU tras un periodo de inactividad. _model_lock protege
        # la carga/descarga y el contador de transcripciones en curso.
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._active = 0
        self._last_used = time.monotonic()
        self._tasks: set[asyncio.Task[None]] = set()

    async def transcribe(
        self,
        audio_queue: asyncio.Queue[AudioChunk],
        text_queue: asyncio.Queue[TranscribedText],
    ) -> None:
        while True:
            chunk = await audio_queue.get()
            task = asyncio.create_task(self._transcribe_one(chunk, text_queue))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            audio_queue.task_done()

    async def monitor_idle(self) -> None:
        """Descarga el modelo de la GPU cuando lleva idle_unload_seconds sin uso.

        Corre en paralelo al pipeline; no hace nada si idle_unload_seconds <= 0.
        """
        if self._config.idle_unload_seconds <= 0:
            return
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(_IDLE_CHECK_INTERVAL_S)
            await loop.run_in_executor(self._executor, self._maybe_unload)

    async def _transcribe_one(
        self,
        chunk: AudioChunk,
        text_queue: asyncio.Queue[TranscribedText],
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            raw_text = await loop.run_in_executor(self._executor, self._run_model, chunk.data)
        except Exception:
            logger.exception('Error transcribing audio chunk seq=%s', chunk.seq)
            return

        if not raw_text.strip():
            logger.info('Discarding empty transcription for seq=%s', chunk.seq)
            return

        await text_queue.put(TranscribedText(seq=chunk.seq, raw=raw_text))

    def _run_model(self, audio: np.ndarray) -> str:
        model = self._acquire_model()
        try:
            segments, _ = model.transcribe(audio, language=self._config.whisper_language)
            return ' '.join(segment.text for segment in segments).strip()
        finally:
            self._release_model()

    def _acquire_model(self) -> Any:
        """Devuelve el modelo cargado, cargándolo bajo demanda, y marca una
        transcripción en curso para que el monitor no lo descargue mientras tanto."""
        with self._model_lock:
            if self._model is None:
                logger.info(
                    'Loading Whisper model %s on %s',
                    self._config.whisper_model,
                    self._config.whisper_device,
                )
                self._model = self._model_factory(
                    self._config.whisper_model,
                    device=self._config.whisper_device,
                    compute_type=self._config.whisper_compute_type,
                )
            self._active += 1
            return self._model

    def _release_model(self) -> None:
        with self._model_lock:
            self._active -= 1
            self._last_used = time.monotonic()

    def _maybe_unload(self) -> bool:
        with self._model_lock:
            idle_for = time.monotonic() - self._last_used
            if (
                self._model is not None
                and self._active == 0
                and idle_for >= self._config.idle_unload_seconds
            ):
                logger.info(
                    'Unloading idle Whisper model after %.0fs of inactivity', idle_for
                )
                self._model = None
                # CTranslate2 libera la VRAM al destruirse el modelo; forzamos el GC
                # para que ocurra ya y no en algún momento indeterminado.
                gc.collect()
                return True
        return False
