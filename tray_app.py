import asyncio
import logging
import sys
import threading
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

from config import load_config
from main import run_pipeline

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    if getattr(sys, 'frozen', False):
        # --noconsole builds have no sys.stdout/stderr to log to; writing
        # there raises AttributeError, so log to a file next to the exe.
        log_path = Path(sys.executable).parent / 'voicecode.log'
        logging.basicConfig(
            level=logging.INFO,
            filename=str(log_path),
            filemode='a',
            format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        )
    else:
        logging.basicConfig(level=logging.INFO)


def _make_icon_image() -> Image.Image:
    image = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((4, 4, 60, 60), fill=(74, 144, 217, 255))
    draw.rounded_rectangle((27, 14, 37, 38), radius=5, fill='white')
    draw.arc((18, 24, 46, 46), start=0, end=180, fill='white', width=3)
    draw.rectangle((30, 44, 34, 50), fill='white')
    return image


class VoiceCodeTrayApp:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._icon: pystray.Icon | None = None

    def _run_pipeline_in_thread(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        config = load_config()
        self._task = self._loop.create_task(run_pipeline(config))
        self._started.set()
        try:
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            logger.info('Pipeline cancelled')
        except Exception:
            logger.exception('Pipeline crashed')
        finally:
            self._loop.close()
            logger.info('Pipeline thread stopped')

    def start_pipeline(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning('Pipeline already running, ignoring start request')
            return
        self._started.clear()
        self._thread = threading.Thread(target=self._run_pipeline_in_thread, daemon=True)
        self._thread.start()

    def stop_pipeline(self) -> None:
        if self._thread is None:
            return
        # Wait for the thread to finish initializing self._loop/self._task
        # before trying to cancel - otherwise a stop requested immediately
        # after start would silently no-op and leave the pipeline running.
        self._started.wait(timeout=5)
        if self._loop is None or self._task is None:
            return
        self._loop.call_soon_threadsafe(self._task.cancel)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _on_restart(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        logger.info('Restart requested from tray menu')
        self.stop_pipeline()
        self.start_pipeline()

    def _on_exit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        logger.info('Exit requested from tray menu')
        self.stop_pipeline()
        icon.stop()

    def run(self) -> None:
        _configure_logging()
        self.start_pipeline()

        menu = pystray.Menu(
            pystray.MenuItem('Reiniciar pipeline', self._on_restart),
            pystray.MenuItem('Salir', self._on_exit),
        )
        self._icon = pystray.Icon('voicecode', _make_icon_image(), 'VoiceCode', menu)
        self._icon.run()


def main() -> None:
    VoiceCodeTrayApp().run()


if __name__ == '__main__':
    main()
