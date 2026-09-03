"""Cross-hardware device detection and tensor migration helpers.

Single source of truth for "what accelerator does this process have
access to, and how do I move a tensor to it?". Used by the
:mod:`qwendopamine.integrations.pytorch.custom_ops` registration to
decide whether to route through CPU or through an accelerator.

Detection order (most preferred first):

    1. CUDA (``torch.cuda.is_available()``)
    2. XPU (``torch.xpu.is_available()``)
    3. MPS (``torch.backends.mps.is_available()``)
    4. Vulkan / Metal / OpenGL via Taichi (``qwendopamine.kernels.taichi.taichi_arch``)
    5. CPU

PyTorch has no first-class Vulkan detection (Vulkan support in
PyTorch is experimental and not exposed through ``is_available()``);
we ask the Taichi runtime directly, because Taichi handles the
CUDA → Vulkan → Metal/OpenGL → CPU fallback internally. This module
treats Taichi as the canonical "what GPU landed" probe.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

import torch

_logger = logging.getLogger(__name__)


_LOCK = threading.RLock()
_DETECTED: list[str] | None = None
_ACTIVE: torch.device | None = None


def _taichi_arch_safe() -> str | None:
    """Return the active Taichi arch string, or None if Taichi is unavailable.

    Wrapped in a try/except because ``taichi_arch()`` may raise when
    Taichi is not installed or fails to initialise (e.g. inside a
    Python REPL on a machine without the Vulkan loader).

    The Taichi runtime is **lazy** — calling ``taichi_arch()``
    triggers a full ``ti.init()`` if it hasn't happened yet, which
    is slow on the first call (Vulkan JIT compiles a runtime
    bitcode). We avoid that cost by checking whether the runtime
    was already initialised.
    """
    try:
        from qwendopamine.kernels.taichi.runtime import _INITIALISED

        if not _INITIALISED:
            # Taichi has not been initialised yet; asking for the
            # arch would trigger the slow first-init path. Defer.
            return None
        from qwendopamine.kernels.taichi import taichi_arch as _impl

        return _impl()
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        _logger.debug("Taichi arch probe failed: %s", exc)
        return None


def detect_available_devices() -> list[str]:
    """Return the list of available accelerator names, ordered by preference.

    The list contains a subset of ``{"cuda", "xpu", "mps", "vulkan", "cpu"}``.
    ``"cpu"`` is always present (the final fallback). The order is the
    preference order: the first element is the one
    :func:`default_device` will pick.

    The result is cached after the first call.
    """
    global _DETECTED
    with _LOCK:
        if _DETECTED is not None:
            return list(_DETECTED)
        available: list[str] = []
        try:
            if torch.cuda.is_available():
                available.append("cuda")
        except Exception as exc:  # noqa: BLE001
            _logger.debug("CUDA probe failed: %s", exc)
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                available.append("xpu")
        except Exception as exc:  # noqa: BLE001
            _logger.debug("XPU probe failed: %s", exc)
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                available.append("mps")
        except Exception as exc:  # noqa: BLE001
            _logger.debug("MPS probe failed: %s", exc)
        # Vulkan / Metal / OpenGL detection: ask Taichi what arch landed.
        taichi_arch = _taichi_arch_safe()
        if taichi_arch is not None:
            taichi_lower = taichi_arch.lower()
            if "vulkan" in taichi_lower and "vulkan" not in available:
                available.append("vulkan")
            elif "metal" in taichi_lower and "metal" not in available:
                available.append("metal")
            elif "opengl" in taichi_lower and "opengl" not in available:
                available.append("opengl")
        available.append("cpu")
        _DETECTED = available
        return list(_DETECTED)


def default_device() -> torch.device:
    """Return the preferred torch device for this process.

    Picks the first element of :func:`detect_available_devices`. Cached.

    **Note:** the underlying device detection does **not** initialise
    the Taichi runtime unless Taichi has already been used elsewhere
    in the process. On a fresh process, the first call returns
    ``torch.device("cpu")`` (because no accelerator has been
    proven). Callers that want GPU routing must invoke
    :func:`qwendopamine.kernels.taichi.taichi_arch` or any Taichi
    op first to land on the active arch.
    """
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None:
            return _ACTIVE
        devices = detect_available_devices()
        head = devices[0]
        if head == "cuda":
            _ACTIVE = torch.device("cuda")
        elif head == "xpu":
            _ACTIVE = torch.device("xpu")
        elif head == "mps":
            _ACTIVE = torch.device("mps")
        elif head in ("vulkan", "metal", "opengl"):
            # PyTorch has no first-class device for these. We still
            # want callers to know "an accelerator is active", so we
            # pick CUDA if available, then CPU. The Taichi kernel
            # copies CPU tensors to the active arch under the hood.
            if torch.cuda.is_available():
                _ACTIVE = torch.device("cuda")
            else:
                _ACTIVE = torch.device("cpu")
        else:
            _ACTIVE = torch.device("cpu")
        return _ACTIVE


def device_for_torch(t: torch.Tensor) -> str:
    """Return the canonical device-type string for a tensor's device.

    ``torch.device("cuda:0").type == "cuda"``; the same for
    ``mps``, ``xpu``, ``cpu``. Vulkan / Metal tensors are not
    representable in PyTorch, so this only ever returns one of the
    four canonical types.
    """
    return t.device.type


def to_active_device(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Move each tensor to :func:`default_device` if it is not already there.

    Returns a list (in the same order) of the migrated tensors. Does
    not modify the input tensors in place.

    Used by the registered custom op kernels: if the caller passed a
    CPU tensor on a system with a CUDA device available, this helper
    migrates it to CUDA before invoking the Taichi kernel, then the
    kernel can run on the GPU.
    """
    target = default_device()
    out: list[torch.Tensor] = []
    for t in tensors:
        if t.device != target:
            out.append(t.to(target))
        else:
            out.append(t)
    return out


def supports_device(device_name: str) -> bool:
    """Return True if the given device name is available on this host."""
    return device_name in detect_available_devices()


def reset_cache() -> None:
    """Reset the device-detection cache.

    Useful in tests that monkey-patch ``torch.cuda.is_available`` or
    the Taichi arch between runs.
    """
    global _DETECTED, _ACTIVE
    with _LOCK:
        _DETECTED = None
        _ACTIVE = None


__all__ = [
    "default_device",
    "detect_available_devices",
    "device_for_torch",
    "reset_cache",
    "supports_device",
    "to_active_device",
]
