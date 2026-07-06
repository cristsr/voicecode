import os
from pathlib import Path

import pytest

import utils.platform as platform_module
from utils.platform import add_nvidia_dll_directories


def test_add_nvidia_dll_directories_registers_bin_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvidia_dir = tmp_path / 'nvidia'
    cublas_bin = nvidia_dir / 'cublas' / 'bin'
    cublas_bin.mkdir(parents=True)
    cudnn_bin = nvidia_dir / 'cudnn' / 'bin'
    cudnn_bin.mkdir(parents=True)
    (nvidia_dir / 'cuda_nvrtc').mkdir()  # no bin/ subfolder - must be skipped safely

    monkeypatch.setattr(platform_module, 'is_windows', lambda: True)
    monkeypatch.setattr(platform_module.site, 'getsitepackages', lambda: [str(tmp_path)])
    monkeypatch.setenv('PATH', r'C:\Windows\System32')

    registered_calls: list[str] = []
    monkeypatch.setattr(os, 'add_dll_directory', registered_calls.append, raising=False)

    result = add_nvidia_dll_directories()

    assert set(result) == {str(cublas_bin), str(cudnn_bin)}
    assert set(registered_calls) == {str(cublas_bin), str(cudnn_bin)}
    path_entries = os.environ['PATH'].split(os.pathsep)
    assert str(cublas_bin) in path_entries
    assert str(cudnn_bin) in path_entries
    assert r'C:\Windows\System32' in path_entries


def test_add_nvidia_dll_directories_does_not_duplicate_path_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvidia_dir = tmp_path / 'nvidia'
    cublas_bin = nvidia_dir / 'cublas' / 'bin'
    cublas_bin.mkdir(parents=True)

    monkeypatch.setattr(platform_module, 'is_windows', lambda: True)
    monkeypatch.setattr(platform_module.site, 'getsitepackages', lambda: [str(tmp_path)])
    monkeypatch.setenv('PATH', str(cublas_bin))
    monkeypatch.setattr(os, 'add_dll_directory', lambda p: None, raising=False)

    add_nvidia_dll_directories()

    path_entries = os.environ['PATH'].split(os.pathsep)
    assert path_entries.count(str(cublas_bin)) == 1


def test_add_nvidia_dll_directories_is_noop_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform_module, 'is_windows', lambda: False)

    result = add_nvidia_dll_directories()

    assert result == []


def test_add_nvidia_dll_directories_handles_missing_nvidia_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform_module, 'is_windows', lambda: True)
    monkeypatch.setattr(platform_module.site, 'getsitepackages', lambda: [str(tmp_path)])

    result = add_nvidia_dll_directories()

    assert result == []


def test_add_nvidia_dll_directories_uses_meipass_when_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PyInstaller --onedir puts collected binaries under dist/<Name>/_internal,
    # which sys._MEIPASS points to - NOT the folder next to the .exe itself.
    internal_dir = tmp_path / 'dist' / 'VoiceCode' / '_internal'
    cublas_bin = internal_dir / 'nvidia' / 'cublas' / 'bin'
    cublas_bin.mkdir(parents=True)

    monkeypatch.setattr(platform_module, 'is_windows', lambda: True)
    monkeypatch.setattr(platform_module.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(platform_module.sys, '_MEIPASS', str(internal_dir), raising=False)
    monkeypatch.setattr(
        platform_module.sys, 'executable', str(tmp_path / 'dist' / 'VoiceCode' / 'VoiceCode.exe')
    )
    monkeypatch.setenv('PATH', r'C:\Windows\System32')
    monkeypatch.setattr(os, 'add_dll_directory', lambda p: None, raising=False)

    # Should not even look at site-packages when frozen.
    monkeypatch.setattr(
        platform_module.site,
        'getsitepackages',
        lambda: (_ for _ in ()).throw(AssertionError('should not be called when frozen')),
    )

    result = add_nvidia_dll_directories()

    assert result == [str(cublas_bin)]


def test_add_nvidia_dll_directories_falls_back_to_exe_dir_when_no_meipass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe_dir = tmp_path / 'dist' / 'VoiceCode'
    cublas_bin = exe_dir / 'nvidia' / 'cublas' / 'bin'
    cublas_bin.mkdir(parents=True)

    monkeypatch.setattr(platform_module, 'is_windows', lambda: True)
    monkeypatch.setattr(platform_module.sys, 'frozen', True, raising=False)
    monkeypatch.delattr(platform_module.sys, '_MEIPASS', raising=False)
    monkeypatch.setattr(platform_module.sys, 'executable', str(exe_dir / 'VoiceCode.exe'))
    monkeypatch.setenv('PATH', r'C:\Windows\System32')
    monkeypatch.setattr(os, 'add_dll_directory', lambda p: None, raising=False)

    result = add_nvidia_dll_directories()

    assert result == [str(cublas_bin)]
