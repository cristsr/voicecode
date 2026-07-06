import asyncio

import pytest

from domain.models import AudioChunk, CleanText, KeyEvent, TranscribedText
from domain.protocols import TranscriberProtocol


@pytest.fixture
def key_queue() -> asyncio.Queue[KeyEvent]:
    return asyncio.Queue()


@pytest.fixture
def audio_queue() -> asyncio.Queue[AudioChunk]:
    return asyncio.Queue()


@pytest.fixture
def text_queue() -> asyncio.Queue[TranscribedText]:
    return asyncio.Queue()


@pytest.fixture
def clean_queue() -> asyncio.Queue[CleanText]:
    return asyncio.Queue()


@pytest.fixture
def mock_transcriber() -> TranscriberProtocol:
    class FakeTranscriber:
        async def transcribe(
            self,
            audio_q: asyncio.Queue[AudioChunk],
            text_q: asyncio.Queue[TranscribedText],
        ) -> None:
            chunk = await audio_q.get()
            await text_q.put(TranscribedText(seq=chunk.seq, raw='texto de prueba'))

    return FakeTranscriber()
