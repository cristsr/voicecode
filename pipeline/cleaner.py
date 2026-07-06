import asyncio
import logging
import re

from config import VoiceCodeConfig
from domain.models import CleanText, TranscribedText

logger = logging.getLogger(__name__)


def clean_text(raw: str, filler_patterns: list[str]) -> str:
    result = raw
    for pattern in filler_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip()
    # Removing a filler word often leaves a dangling comma/semicolon/colon
    # behind (e.g. "eh, quiero" -> ", quiero"); strip it before capitalizing.
    result = re.sub(r'^[,;:]+\s*', '', result)
    if result:
        result = result[0].upper() + result[1:]
    return result


class RegexCleaner:
    def __init__(self, config: VoiceCodeConfig) -> None:
        self._config = config

    async def clean(
        self,
        text_queue: asyncio.Queue[TranscribedText],
        clean_queue: asyncio.Queue[CleanText],
    ) -> None:
        while True:
            item = await text_queue.get()
            text = clean_text(item.raw, self._config.filler_patterns)
            await clean_queue.put(CleanText(seq=item.seq, text=text))
            text_queue.task_done()
