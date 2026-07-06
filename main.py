import asyncio
import logging

from config import VoiceCodeConfig, load_config
from domain.models import AudioChunk, CleanText, KeyEvent, TranscribedText
from pipeline.cleaner import RegexCleaner
from pipeline.listener import PynputListener
from pipeline.recorder import SoundDeviceRecorder
from pipeline.transcriber import WhisperTranscriber
from pipeline.writer import ClipboardWriter

logger = logging.getLogger(__name__)


async def run_pipeline(config: VoiceCodeConfig) -> None:
    key_queue: asyncio.Queue[KeyEvent] = asyncio.Queue()
    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
    text_queue: asyncio.Queue[TranscribedText] = asyncio.Queue()
    clean_queue: asyncio.Queue[CleanText] = asyncio.Queue()

    listener = PynputListener(config)
    recorder = SoundDeviceRecorder(config)
    transcriber = WhisperTranscriber(config)
    cleaner = RegexCleaner(config)
    writer = ClipboardWriter(config)

    logger.info('VoiceCode started - hold %s to talk', config.ptt_key)

    await asyncio.gather(
        listener.listen(key_queue),
        recorder.record(key_queue, audio_queue),
        transcriber.transcribe(audio_queue, text_queue),
        transcriber.monitor_idle(),
        cleaner.clean(text_queue, clean_queue),
        writer.write(clean_queue),
    )


async def main() -> None:
    config = load_config()
    await run_pipeline(config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
