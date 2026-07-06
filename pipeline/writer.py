import asyncio
import logging

import pyperclip
from pynput.keyboard import Controller, Key

from config import VoiceCodeConfig
from domain.models import CleanText

logger = logging.getLogger(__name__)


class SequenceBuffer:
    """Buffers out-of-order CleanText items and releases them in seq order."""

    def __init__(self) -> None:
        self.expected_seq = 0
        self.pending: dict[int, CleanText] = {}

    def process(self, item: CleanText) -> list[CleanText]:
        ready: list[CleanText] = []
        if item.seq == self.expected_seq:
            ready.append(item)
            self.expected_seq += 1
            while self.expected_seq in self.pending:
                ready.append(self.pending.pop(self.expected_seq))
                self.expected_seq += 1
        else:
            self.pending[item.seq] = item
        return ready


class ClipboardWriter:
    def __init__(self, config: VoiceCodeConfig) -> None:
        self._config = config
        self._buffer = SequenceBuffer()
        self._keyboard = Controller()

    async def write(self, clean_queue: asyncio.Queue[CleanText]) -> None:
        while True:
            item = await clean_queue.get()
            for ready_item in self._buffer.process(item):
                if ready_item.text:
                    await self._emit(ready_item)
                else:
                    logger.info('Skipping empty clean text for seq=%s', ready_item.seq)
            clean_queue.task_done()

    async def _emit(self, item: CleanText) -> None:
        backup = pyperclip.paste()
        try:
            pyperclip.copy(item.text)
            self._keyboard.press(Key.ctrl)
            self._keyboard.press('v')
            self._keyboard.release('v')
            self._keyboard.release(Key.ctrl)
            await asyncio.sleep(self._config.clipboard_restore_delay_ms / 1000)
        finally:
            pyperclip.copy(backup)
