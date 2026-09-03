"""Lazy Taichi runtime initialisation.

Resolves the active Taichi architecture (CUDA GPU if available, otherwise CPU)
and exposes it via :func:`taichi_arch`. Taichi handles JIT compilation of
``@ti.kernel`` functions and dispatches them to the resolved backend, so the
same kernel source produces CPU or GPU code with no CPU-only fallback path
required.
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
    """Initialise the Taichi runtime with auto-selected architecture.

    Passes an explicit preference list ``[ti.cuda, ti.gpu, ti.cpu]`` to
    :func:`ti.init` so Taichi picks the first backend that actually
    works on this machine (CUDA > generic GPU > CPU). When no
    preference is given, Taichi's own default priority applies
    (CUDA if available, then Vulkan/Metal, then CPU).

    The chosen arch is stored in the module-level ``_ARCH`` variable
    so every subsequent kernel compilation picks the same target.
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
                arch=[  # type: ignore[missing-attribute]
                    ti.cuda,  # type: ignore[missing-attribute]
                    ti.gpu,  # type: ignore[missing-attribute]
                    ti.cpu,  # type: ignore[missing-attribute]
                ],
                default_fp=ti.f32,  # type: ignore[missing-attribute]
            )
            resolved = ti.cfg.arch  # type: ignore[attr-defined]
            chosen = str(resolved) if resolved is not None else "auto"
        except Exception:  # noqa: BLE001
            # Last-resort fallback: bare ``ti.init()`` lets Taichi apply
            # its built-in default priority (CUDA > Vulkan/Metal > CPU).
            ti.init(default_fp=ti.f32)  # type: ignore[missing-attribute]
            resolved = ti.cfg.arch  # type: ignore[attr-defined]
            chosen = str(resolved) if resolved is not None else "auto"

        _ARCH = chosen
        _INITIALISED = True


def is_available() -> bool:
    """Return True if the Taichi runtime could be initialised."""
    _initialise()
    return _HAS_TAICHI and _ARCH != "unavailable"


def taichi_arch() -> str:
    """Return the active Taichi backend (``cpu`` / ``cuda`` / ``gpu``)."""
    _initialise()
    return _ARCH


def require() -> Any:
    """Return the imported ``taichi`` module, raising a clear error if missing."""
    _initialise()
    if not _HAS_TAICHI:
        raise RuntimeError(
            "Taichi is required for the GDN-2 Taichi backend. "
            f"Original import error: {_IMPORT_ERROR!r}"
        )
    return ti


__all__ = ["is_available", "require", "taichi_arch"]
