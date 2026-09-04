"""Lazy Taichi runtime initialisation.

Lets Taichi pick the backend itself: ``ti.init(arch=ti.gpu)`` makes Taichi
try CUDA → Vulkan → Metal/OpenGL → CPU in that order. We do not steer
the choice manually; the resolved backend is read back via
:func:`taichi_arch` so callers can report what landed.
"""

from __future__ import annotations

import threading
from typing import Any

try:
    import taichi as ti
except (ImportError, RuntimeError) as exc:  # pragma: no cover - guarded import
    _IMPORT_ERROR = exc
    ti = None  # type: ignore[assignment]
    _HAS_TAICHI = False
else:
    _IMPORT_ERROR = None
    _HAS_TAICHI = True


_LOCK = threading.Lock()
_INITIALISED = False
_ARCH: str = "cpu"


def _initialise() -> None:
    """Initialise the Taichi runtime, deferring backend choice to Taichi.

    ``ti.init(arch=ti.gpu)`` asks Taichi to try CUDA, then Vulkan, then
    Metal/OpenGL, then fall back to CPU. If even ``ti.gpu`` cannot be
    resolved (e.g. on a headless box with no GPU drivers), a second
    ``ti.init()`` call with no arguments applies Taichi's own default
    priority (CUDA > Vulkan/Metal > CPU).

    The chosen arch is stored in ``_ARCH`` so every subsequent kernel
    compilation picks the same target.
    """
    global _INITIALISED, _ARCH
    with _LOCK:
        if _INITIALISED:
            return
        if not _HAS_TAICHI:
            _INITIALISED = True
            _ARCH = "unavailable"
            return

        chosen: str = "cpu"
        try:
            ti.init(  # type: ignore[missing-attribute]
                arch=ti.gpu,  # type: ignore[missing-attribute]
                default_fp=ti.f32,  # type: ignore[missing-attribute]
            )
            resolved = ti.cfg.arch  # type: ignore[attr-defined]
            chosen = str(resolved) if resolved is not None else "auto"
        except Exception:  # noqa: BLE001
            ti.init(default_fp=ti.f32)  # type: ignore[missing-attribute]
            resolved = ti.cfg.arch  # type: ignore[attr-defined]
            chosen = str(resolved) if resolved is not None else "auto"

        _ARCH = chosen
        _INITIALISED = True


def is_available(initialize: bool = True) -> bool:
    r"""is_available(initialize: bool = True) -> bool

    Return True if the Taichi runtime could be initialised.

    Args:
        initialize (bool): When True (default), lazily initialise the
            Taichi runtime if it has not been initialised yet. When
            False, return the cached availability without
            initialising. This is the safe query for callers that
            only want to *probe* without paying the initialisation
            cost (e.g. tests that monkey-patch the arch after the
            first call).

    Returns:
        bool: ``True`` when Taichi imported and a backend was selected,
        ``False`` otherwise.
    """
    if initialize:
        _initialise()
    return _HAS_TAICHI and _ARCH != "unavailable"


def taichi_arch(initialize: bool = True) -> str:
    r"""taichi_arch(initialize: bool = True) -> str

    Return the active Taichi backend string.

    Args:
        initialize (bool): When True (default), lazily initialise if
            needed. When False, return the cached arch without
            triggering a Taichi ``init`` call.

    Returns:
        str: Resolved backend (``"cpu"``, ``"cuda"``, ``"gpu"``, or
        ``"unavailable"``).
    """
    if initialize:
        _initialise()
    return _ARCH


def require() -> Any:
    r"""require() -> Any

    Return the imported ``taichi`` module, raising a clear error if missing.

    Returns:
        Any: The ``taichi`` module.

    Raises:
        RuntimeError: If Taichi could not be imported or initialised.
    """
    _initialise()
    if not _HAS_TAICHI:
        raise RuntimeError(
            "Taichi is required for the GDN-2 Taichi backend. "
            f"Original import error: {_IMPORT_ERROR!r}"
        )
    return ti


__all__ = ["is_available", "require", "taichi_arch"]
