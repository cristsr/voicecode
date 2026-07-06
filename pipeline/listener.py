import asyncio
import logging

from pynput import keyboard
from pynput.keyboard import Key

from config import VoiceCodeConfig
from domain.models import KeyEvent

logger = logging.getLogger(__name__)


class PynputListener:
    def __init__(self, config: VoiceCodeConfig) -> None:
        self._config = config
        self._target_key = getattr(Key, config.ptt_key)

    async def listen(self, key_queue: asyncio.Queue[KeyEvent]) -> None:
        loop = asyncio.get_running_loop()

        def on_press(key: object) -> None:
            try:
                if key == self._target_key:
                    loop.call_soon_threadsafe(
                        key_queue.put_nowait,
                        KeyEvent(kind='down', key=self._config.ptt_key),
                    )
            except Exception:
                logger.exception('Error handling key press event')

        def on_release(key: object) -> None:
            try:
                if key == self._target_key:
                    loop.call_soon_threadsafe(
                        key_queue.put_nowait,
                        KeyEvent(kind='up', key=self._config.ptt_key),
                    )
            except Exception:
                logger.exception('Error handling key release event')

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        try:
            await asyncio.Event().wait()
        finally:
            listener.stop()
