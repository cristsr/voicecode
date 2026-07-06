import asyncio
from unittest.mock import AsyncMock

import pytest

from config import VoiceCodeConfig
from domain.models import CleanText
from pipeline.writer import ClipboardWriter


@pytest.mark.asyncio
async def test_skips_emit_for_empty_text_but_advances_sequence() -> None:
    config = VoiceCodeConfig()
    writer = ClipboardWriter(config)
    writer._emit = AsyncMock()  # type: ignore[method-assign]
    clean_queue: asyncio.Queue[CleanText] = asyncio.Queue()

    task = asyncio.create_task(writer.write(clean_queue))

    await clean_queue.put(CleanText(seq=0, text=''))
    await clean_queue.put(CleanText(seq=1, text='hola'))
    await clean_queue.join()
    task.cancel()

    writer._emit.assert_called_once()
    (emitted_item,) = writer._emit.call_args.args
    assert emitted_item == CleanText(seq=1, text='hola')
    assert writer._buffer.expected_seq == 2


@pytest.mark.asyncio
async def test_skips_emit_for_all_items_when_all_are_empty() -> None:
    config = VoiceCodeConfig()
    writer = ClipboardWriter(config)
    writer._emit = AsyncMock()  # type: ignore[method-assign]
    clean_queue: asyncio.Queue[CleanText] = asyncio.Queue()

    task = asyncio.create_task(writer.write(clean_queue))

    await clean_queue.put(CleanText(seq=0, text=''))
    await clean_queue.join()
    task.cancel()

    writer._emit.assert_not_called()
    assert writer._buffer.expected_seq == 1
