from pathlib import Path

from config import VoiceCodeConfig, load_config


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / 'does_not_exist.toml')

    assert config == VoiceCodeConfig()


def test_loads_all_values_from_toml_file(tmp_path: Path) -> None:
    toml_path = tmp_path / 'config.toml'
    toml_path.write_text(
        """
        [ptt]
        key = "f9"

        [audio]
        sample_rate = 8000
        channels = 2
        min_audio_duration_ms = 500

        [whisper]
        model = "small"
        device = "cpu"
        compute_type = "int8"
        language = "en"

        [transcriber]
        max_workers = 4

        [cleaner]
        filler_patterns = ["\\\\beh+\\\\b"]

        [writer]
        clipboard_restore_delay_ms = 100
        """,
        encoding='utf-8',
    )

    config = load_config(toml_path)

    assert config == VoiceCodeConfig(
        ptt_key='f9',
        sample_rate=8000,
        channels=2,
        min_audio_duration_ms=500,
        whisper_model='small',
        whisper_device='cpu',
        whisper_compute_type='int8',
        whisper_language='en',
        transcriber_max_workers=4,
        filler_patterns=[r'\beh+\b'],
        clipboard_restore_delay_ms=100,
    )


def test_partial_toml_keeps_defaults_for_missing_fields(tmp_path: Path) -> None:
    toml_path = tmp_path / 'config.toml'
    toml_path.write_text(
        """
        [ptt]
        key = "f9"

        [whisper]
        language = "en"
        """,
        encoding='utf-8',
    )

    config = load_config(toml_path)
    defaults = VoiceCodeConfig()

    assert config.ptt_key == 'f9'
    assert config.whisper_language == 'en'
    assert config.sample_rate == defaults.sample_rate
    assert config.whisper_model == defaults.whisper_model
    assert config.filler_patterns == defaults.filler_patterns


def test_missing_sections_keep_defaults(tmp_path: Path) -> None:
    toml_path = tmp_path / 'config.toml'
    toml_path.write_text('[audio]\nsample_rate = 44100\n', encoding='utf-8')

    config = load_config(toml_path)

    assert config.sample_rate == 44100
    assert config.channels == VoiceCodeConfig().channels
    assert config.ptt_key == VoiceCodeConfig().ptt_key


def test_default_config_toml_file_loads_without_error() -> None:
    config = load_config()

    assert config == VoiceCodeConfig()
