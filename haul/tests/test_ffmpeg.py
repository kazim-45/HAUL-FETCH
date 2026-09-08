"""Unit tests for haul.core.ffmpeg — the bundled-binary-first,
system-PATH-fallback resolution logic.

resolve_ffmpeg_path() is cached with functools.lru_cache, so every
test here clears the cache first (via monkeypatch's built-in
teardown undoing our patches doesn't clear the cache automatically —
we clear it explicitly) to get a clean read on each scenario.
"""

from __future__ import annotations

import sys
import types

import haul.core.ffmpeg as ffmpeg_module


def _install_fake_imageio_ffmpeg(monkeypatch, *, path=None, raises=False):
    """Installs a fake `imageio_ffmpeg` module in sys.modules for the
    duration of one test, either returning `path` from
    get_ffmpeg_exe() or raising to simulate it being unable to
    resolve a binary for this platform."""
    fake = types.ModuleType("imageio_ffmpeg")

    def get_ffmpeg_exe():
        if raises:
            raise RuntimeError("no ffmpeg build available for this platform")
        return path

    fake.get_ffmpeg_exe = get_ffmpeg_exe
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)


def test_prefers_bundled_binary_when_available(monkeypatch):
    ffmpeg_module.resolve_ffmpeg_path.cache_clear()
    _install_fake_imageio_ffmpeg(monkeypatch, path="/fake/bundled/ffmpeg")

    assert ffmpeg_module.resolve_ffmpeg_path() == "/fake/bundled/ffmpeg"
    assert ffmpeg_module.ffmpeg_available() is True


def test_falls_back_to_system_path_when_bundle_unavailable(monkeypatch):
    ffmpeg_module.resolve_ffmpeg_path.cache_clear()
    _install_fake_imageio_ffmpeg(monkeypatch, raises=True)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    assert ffmpeg_module.resolve_ffmpeg_path() == "/usr/bin/ffmpeg"


def test_falls_back_to_system_path_when_bundle_not_installed(monkeypatch):
    ffmpeg_module.resolve_ffmpeg_path.cache_clear()
    monkeypatch.delitem(sys.modules, "imageio_ffmpeg", raising=False)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)
    # Simulate imageio_ffmpeg genuinely not being importable, since it
    # isn't present as a real package in this environment either.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "imageio_ffmpeg":
            raise ImportError("No module named 'imageio_ffmpeg'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert ffmpeg_module.resolve_ffmpeg_path() is None
    assert ffmpeg_module.ffmpeg_available() is False


def test_returns_none_when_neither_bundle_nor_system_available(monkeypatch):
    ffmpeg_module.resolve_ffmpeg_path.cache_clear()
    _install_fake_imageio_ffmpeg(monkeypatch, raises=True)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)

    assert ffmpeg_module.resolve_ffmpeg_path() is None
    assert ffmpeg_module.ffmpeg_available() is False


def test_result_is_cached_across_calls(monkeypatch):
    ffmpeg_module.resolve_ffmpeg_path.cache_clear()
    calls = []

    fake = types.ModuleType("imageio_ffmpeg")

    def get_ffmpeg_exe():
        calls.append(1)
        return "/fake/bundled/ffmpeg"

    fake.get_ffmpeg_exe = get_ffmpeg_exe
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)

    ffmpeg_module.resolve_ffmpeg_path()
    ffmpeg_module.resolve_ffmpeg_path()
    ffmpeg_module.resolve_ffmpeg_path()

    assert len(calls) == 1
