import asyncio

import numpy as np
import pytest

from config import VoiceCodeConfig
from domain.models import AudioChunk, CleanText, TranscribedText
from domain.protocols import TranscriberProtocol
from pipeline.cleaner import RegexCleaner
from pipeline.writer import SequenceBuffer


@pytest.mark.asyncio
async def test_synthetic_audio_chunks_emit_in_order(
    mock_transcriber: TranscriberProtocol,
) -> None:
    config = VoiceCodeConfig()
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
    text_queue: asyncio.Queue[TranscribedText] = asyncio.Queue()
    clean_queue: asyncio.Queue[CleanText] = asyncio.Queue()

    cleaner = RegexCleaner(config)
    buffer = SequenceBuffer()

    for seq in range(3):
        await audio_queue.put(
            AudioChunk(seq=seq, data=np.zeros(1600, dtype=np.float32), sample_rate=16000)
        )

    transcriber_tasks = [
        asyncio.create_task(mock_transcriber.transcribe(audio_queue, text_queue))
        for _ in range(3)
    ]
    cleaner_task = asyncio.create_task(cleaner.clean(text_queue, clean_queue))

    emitted: list[CleanText] = []
    for _ in range(3):
        item = await asyncio.wait_for(clean_queue.get(), timeout=1)
        emitted.extend(buffer.process(item))

    for task in transcriber_tasks:
        task.cancel()
    cleaner_task.cancel()

    assert [item.seq for item in emitted] == [0, 1, 2]
    assert all(item.text == 'Texto de prueba' for item in emitted)
