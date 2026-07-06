import asyncio
from typing import Protocol

from domain.models import AudioChunk, CleanText, KeyEvent, TranscribedText


class KeyListenerProtocol(Protocol):
    async def listen(self, key_queue: asyncio.Queue[KeyEvent]) -> None: ...


class AudioRecorderProtocol(Protocol):
    async def record(
        self,
        key_queue: asyncio.Queue[KeyEvent],
        audio_queue: asyncio.Queue[AudioChunk],
    ) -> None: ...


class TranscriberProtocol(Protocol):
    async def transcribe(
        self,
        audio_queue: asyncio.Queue[AudioChunk],
        text_queue: asyncio.Queue[TranscribedText],
    ) -> None: ...


class CleanerProtocol(Protocol):
    async def clean(
        self,
        text_queue: asyncio.Queue[TranscribedText],
        clean_queue: asyncio.Queue[CleanText],
    ) -> None: ...


class WriterProtocol(Protocol):
    async def write(self, clean_queue: asyncio.Queue[CleanText]) -> None: ...
