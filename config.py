import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _default_config_path() -> Path:
    # When packaged with PyInstaller, __file__ resolves inside the frozen
    # temp bundle (read-only, wiped on exit) - the editable config.toml
    # must instead live next to the .exe on disk.
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent / 'config.toml'
    return Path(__file__).parent / 'config.toml'


DEFAULT_CONFIG_PATH = _default_config_path()

# Maps [toml section][toml key] -> VoiceCodeConfig field name
_FIELD_MAP: dict[str, dict[str, str]] = {
    'ptt': {
        'key': 'ptt_key',
    },
    'audio': {
        'sample_rate': 'sample_rate',
        'channels': 'channels',
        'min_audio_duration_ms': 'min_audio_duration_ms',
    },
    'whisper': {
        'model': 'whisper_model',
        'device': 'whisper_device',
        'compute_type': 'whisper_compute_type',
        'language': 'whisper_language',
    },
    'transcriber': {
        'max_workers': 'transcriber_max_workers',
        'idle_unload_seconds': 'idle_unload_seconds',
    },
    'cleaner': {
        'filler_patterns': 'filler_patterns',
    },
    'writer': {
        'clipboard_restore_delay_ms': 'clipboard_restore_delay_ms',
    },
}


@dataclass
class VoiceCodeConfig:
    # PTT
    ptt_key: str = 'f12'

    # Audio
    sample_rate: int = 16000
    channels: int = 1
    min_audio_duration_ms: int = 300

    # Whisper
    whisper_model: str = 'large-v3'
    whisper_device: str = 'cuda'
    whisper_compute_type: str = 'float16'
    whisper_language: str = 'es'

    # Transcriber
    transcriber_max_workers: int = 2
    # Descarga el modelo de la GPU tras N segundos sin uso (0 = nunca descargar).
    idle_unload_seconds: int = 300

    # Cleaner - filler regex patterns
    filler_patterns: list[str] = field(default_factory=lambda: [
        r'\beh+\b',
        r'\bmmm+\b',
        r'\bo sea\b',
        r'\bdigamos\b',
        r'\bbásicamente\b',
        r'\bpues\b',
        r'\benton?ces\b',
        r'\bla verdad\b',
    ])

    # Writer
    clipboard_restore_delay_ms: int = 50


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> VoiceCodeConfig:
    """Load VoiceCodeConfig from a TOML file, falling back to defaults for
    any section/key that is missing or if the file itself does not exist."""
    path = Path(path)
    if not path.exists():
        return VoiceCodeConfig()

    with path.open('rb') as toml_file:
        data = tomllib.load(toml_file)

    kwargs: dict[str, object] = {}
    for section, keys in _FIELD_MAP.items():
        section_data = data.get(section, {})
        for toml_key, field_name in keys.items():
            if toml_key in section_data:
                kwargs[field_name] = section_data[toml_key]

    return VoiceCodeConfig(**kwargs)
