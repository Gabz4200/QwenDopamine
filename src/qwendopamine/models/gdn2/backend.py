# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Backend resolution and execution dispatch for GDN-2.

This module selects the concrete GDN-2 execution backend based on the runtime
environment (CPU vs CUDA, training vs inference, sequence length) and the
user-requested backend string.
"""

from __future__ import annotations

# Module-level single-warning guard for CPU fallback
_WARNED_FALLBACKS: set[str] = set()


def _warn_fallback_once(reason: str) -> None:
    if reason not in _WARNED_FALLBACKS:
        _WARNED_FALLBACKS.add(reason)
        import warnings

        warnings.warn(f"[gdn2] Using pure PyTorch fallback: {reason}", stacklevel=3)


GDN2_BACKENDS = (
    "auto",
    "taichi",
    "torch",
    "torch-chunk",
    "torch-recurrent",
    "compiled",
    "triton",
    "fla",
)

_SINGLE_TOKEN_SEQ_LEN = 1
_RECURRENT_SHORT_SEQ_LEN = 64

_DEFAULT_CHUNK_SIZE = 64
_DEFAULT_BACKEND = "auto"
_DEFAULT_COMPILE_BACKEND = False


def _taichi_ok() -> bool:
    try:
        from qwendopamine.kernels.taichi import is_available

        return bool(is_available())
    except (ImportError, RuntimeError):
        return False


def resolve_gdn2_backend(
    requested: str,
    *,
    training: bool,
    seq_len: int,
) -> str:
    r"""Resolve the concrete GDN-2 execution backend for a forward call.

    The Taichi backend is the single hardware-accelerated engine; Taichi
    itself picks CUDA → Vulkan → Metal/OpenGL → CPU. ``"auto"`` selects
    Taichi when available and otherwise falls back to the chunkwise /
    recurrent pure-PyTorch reference kernels.
    """
    if requested not in GDN2_BACKENDS:
        raise ValueError(
            f"Invalid GDN-2 backend '{requested}'. Valid backends: {list(GDN2_BACKENDS)}"
        )
    if requested != "auto":
        # The CUDA-bound triton/fla paths were replaced by Taichi; the
        # old scalar names now route to the equivalent path.
        if requested in ("triton", "fla"):
            return "taichi" if _taichi_ok() else "torch-chunk"
        if requested == "compiled":
            return "torch-chunk"
        return requested

    if _taichi_ok():
        return "taichi"

    if not training and seq_len <= _SINGLE_TOKEN_SEQ_LEN:
        return "torch-recurrent"
    if training:
        return "torch-chunk"
    if seq_len <= _RECURRENT_SHORT_SEQ_LEN:
        return "torch-recurrent"
    return "torch-chunk"


__all__ = [
    "GDN2_BACKENDS",
    "resolve_gdn2_backend",
]
