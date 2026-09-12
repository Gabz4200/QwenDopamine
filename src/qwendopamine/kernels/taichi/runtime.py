"""Lazy Taichi runtime initialisation.

Lets Taichi pick the backend itself: ``ti.init(arch=ti.gpu)`` makes Taichi
try CUDA → Vulkan → Metal/OpenGL → CPU in that order. We do not steer
the choice manually; the resolved backend is read back via
:fun:`taichi_arch` so callers can report what landed.
"""

from __future__ import annotations

import threading
from typing import Any

import taichi as ti

_LOCK = threading.Lock()
_INITIALISED = False
_ARCH: str = "cpu"


def _initialise() -> None:
    """Initialise the Taichi runtime, deferring backend choice to Taichi.

    ``ti.init(arch=ti.gpu)`` asks Taichi to try CUDA, then Vulkan, then
    Metal/OpenGL, then CPU as a last resort.

    The chosen arch is stored in ``_ARCH`` so every subsequent kernel
    compilation picks the same target.
    """
    global _INITIALISED, _ARCH
    with _LOCK:
        if _INITIALISED:
            return
        ti.init(  # type: ignore[missing- attribute]
            arch=ti.gpu,  # type: ignore[missing-attribute]
            default_fp=ti.f32,  # type: ignore[missing-attribute]
        )
        resolved = ti.cfg.arch  # type: ignore[attr-defined]
        chosen = str(resolved) if resolved is not None else "auto"

        _ARCH = chosen
        _INITIALISED = True


def is_available(initialize: bool = True) -> bool:
    r"""is_available(initialize: bool = True) -> bool

    Return True if the Taichi runtime is available.

    Args:
        initialize (bool): When True (default), lazily initialise the
            Taichi runtime if it has not been initialised yet. When
            False, return True without initialising. This is the safe
            query for callers that only want to *probe* without paying
            the initialisation cost (e.g. tests that monkey-patch the
            arch after the first call).

    Returns:
        bool: Always ``True`` — Taichi is a hard dependency.
    """
    if initialize:
        _initialise()
    return True


def taichi_arch(initialize: bool = True) -> str:
    r"""taichi_arch(initialize: bool = True) -> str

    Return the active Taichi backend string.

    Args:
        initialize (bool): When True (default), lazily initialise if
            needed. When False, return the cached arch without
            triggering a Taichi ``init`` call.

    Returns:
        str: Resolved backend (e.g. ``"cpu"``, ``"cuda"``, ``"gpu"``).
    """
    if initialize:
        _initialise()
    return _ARCH


def require() -> Any:
    r"""require() -> Any

    Return the imported ``taichi`` module.

    Returns:
        Any: The ``taichi`` module.
    """
    _initialise()
    return ti


__all__ = ["is_available", "require", "taichi_arch"]
