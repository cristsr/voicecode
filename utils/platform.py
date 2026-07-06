import logging
import os
import platform
import shutil
import site
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_os_name() -> str:
    return platform.system()


def is_windows() -> bool:
    return get_os_name() == 'Windows'


def is_linux() -> bool:
    return get_os_name() == 'Linux'


def is_wayland() -> bool:
    return is_linux() and os.environ.get('XDG_SESSION_TYPE') == 'wayland'


def check_paste_dependencies() -> list[str]:
    """Return a list of human-readable warnings about missing OS-level
    dependencies required to simulate Ctrl+V on the current platform."""
    warnings: list[str] = []
    if is_wayland() and shutil.which('ydotool') is None:
        warnings.append(
            'Linux Wayland detected but ydotool is not installed - '
            'key simulation (Ctrl+V paste) will not work.'
        )
    return warnings


def _nvidia_search_roots() -> list[Path]:
    # When packaged with PyInstaller, the nvidia DLL folders collected via
    # `--collect-binaries` end up under sys._MEIPASS - for --onedir builds
    # that is dist/<Name>/_internal (not next to the .exe itself), and for
    # --onefile builds it's the temp extraction directory. site-packages
    # does not exist in a frozen build, so it would find nothing there.
    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            return [Path(meipass)]
        return [Path(sys.executable).parent]
    return [Path(p) for p in site.getsitepackages()]


def add_nvidia_dll_directories() -> list[str]:
    """On Windows, register the bin/ directories of the nvidia-cublas-cu12 /
    nvidia-cudnn-cu12 packages so ctranslate2 can find their DLLs at
    inference time.

    Unlike Linux, Windows does not search inside arbitrary site-packages
    folders for shared libraries - without this, ctranslate2 fails with
    "Library cublas64_12.dll is not found" even though the package is
    installed. os.add_dll_directory() alone is not enough here: ctranslate2's
    native DLL loading does not consistently honor AddDllDirectory-registered
    paths, so the directories are also prepended to PATH, which every
    LoadLibrary call respects regardless of how the caller was built.
    Returns the directories that were registered.
    """
    if not is_windows():
        return []

    roots = _nvidia_search_roots()
    logger.info('add_nvidia_dll_directories: frozen=%s search_roots=%s', getattr(sys, 'frozen', False), roots)

    added: list[str] = []
    for root in roots:
        nvidia_dir = root / 'nvidia'
        if not nvidia_dir.is_dir():
            logger.info('add_nvidia_dll_directories: %s does not exist, skipping', nvidia_dir)
            continue
        for package_dir in nvidia_dir.iterdir():
            bin_dir = package_dir / 'bin'
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                added.append(str(bin_dir))

    if added:
        existing_path = os.environ.get('PATH', '')
        existing_entries = existing_path.split(os.pathsep) if existing_path else []
        new_entries = [d for d in added if d not in existing_entries]
        if new_entries:
            os.environ['PATH'] = os.pathsep.join(new_entries + existing_entries)

    logger.info('add_nvidia_dll_directories: registered=%s', added)
    return added
