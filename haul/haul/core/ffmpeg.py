"""Resolves a path to a working FFmpeg binary without requiring the
user to install anything manually.

HAUL depends on **imageio-ffmpeg**, a PyPI package that ships a real,
statically-linked FFmpeg build inside platform-specific wheels
(Windows, macOS Intel/ARM, Linux x86_64). A plain ``pip install
haul-cli`` already pulls in the right binary for the current
platform as an ordinary dependency, the same way it pulls in
``requests`` or ``rich`` — there's no separate download step, no
admin rights needed, and nothing to add to PATH by hand.

If that ever fails to resolve — an unsupported platform, a
``--no-deps`` install, an unusual environment — HAUL falls back to
whatever ``ffmpeg`` it finds on the system PATH, so a regular system
install (``apt install ffmpeg``, ``brew install ffmpeg``, ...) still
works exactly as it did before. Every caller in this package goes
through :func:`resolve_ffmpeg_path` / :func:`ffmpeg_available` rather
than hardcoding the string ``"ffmpeg"``, so this is the only place
that decision is made.
"""

from __future__ import annotations

import functools
import shutil


@functools.lru_cache(maxsize=1)
def resolve_ffmpeg_path() -> str | None:
    """Returns an absolute path to a usable FFmpeg binary, or None if
    none could be found. Cached after the first successful (or
    failed) lookup, since resolving the bundled binary's path is not
    free and this is called on every merge/audio-extract."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # Either imageio-ffmpeg isn't installed, or it couldn't
        # locate/provide a binary for this platform. Fall back to a
        # regular system install rather than failing outright.
        return shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    return resolve_ffmpeg_path() is not None
