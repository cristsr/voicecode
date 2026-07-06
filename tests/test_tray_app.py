import asyncio

import pytest

import tray_app
from config import VoiceCodeConfig
from tray_app import VoiceCodeTrayApp


async def _fake_run_pipeline_forever(config: VoiceCodeConfig) -> None:
    await asyncio.Event().wait()


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tray_app, 'run_pipeline', _fake_run_pipeline_forever)
    monkeypatch.setattr(tray_app, 'load_config', lambda: VoiceCodeConfig())


def _wait_until_started(app: VoiceCodeTrayApp) -> None:
    if not app._started.wait(timeout=1):
        raise TimeoutError('Pipeline thread never finished initializing')


def test_start_pipeline_runs_in_background_thread() -> None:
    app = VoiceCodeTrayApp()
    app.start_pipeline()
    _wait_until_started(app)

    assert app._thread.is_alive()

    app.stop_pipeline()


def test_stop_pipeline_cancels_task_and_joins_thread() -> None:
    app = VoiceCodeTrayApp()
    app.start_pipeline()
    _wait_until_started(app)

    app.stop_pipeline()

    assert not app._thread.is_alive()


def test_start_pipeline_ignores_duplicate_start_while_running() -> None:
    app = VoiceCodeTrayApp()
    app.start_pipeline()
    _wait_until_started(app)
    first_thread = app._thread

    app.start_pipeline()

    assert app._thread is first_thread

    app.stop_pipeline()


def test_restart_stops_old_thread_and_starts_new_one() -> None:
    app = VoiceCodeTrayApp()
    app.start_pipeline()
    _wait_until_started(app)
    first_thread = app._thread

    app._on_restart(icon=None, item=None)
    _wait_until_started(app)

    assert app._thread is not first_thread
    assert not first_thread.is_alive()
    assert app._thread.is_alive()

    app.stop_pipeline()


def test_stop_pipeline_called_immediately_after_start_does_not_leave_it_running() -> None:
    app = VoiceCodeTrayApp()
    app.start_pipeline()
    app.stop_pipeline()  # no wait in between - exercises the _started guard

    assert not app._thread.is_alive()
