# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Backend resolution and execution dispatch for GDN-2.

This module selects the concrete GDN-2 execution backend based on the runtime
environment (CPU vs CUDA, training vs inference, sequence length) and the
user-requested backend string.
"""

from __future__ import annotations

import torch

# Module-level single-warning guard for CPU fallback
_WARNED_FALLBACKS: set[str] = set()


def _warn_fallback_once(reason: str) -> None:
    if reason not in _WARNED_FALLBACKS:
        _WARNED_FALLBACKS.add(reason)
        import warnings

        warnings.warn(f"[gdn2] Using pure PyTorch fallback: {reason}", stacklevel=3)


GDN2_BACKENDS = (
    "auto",
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


def resolve_gdn2_backend(
    requested: str,
    *,
    training: bool,
    seq_len: int,
) -> str:
    r"""Resolve the concrete GDN-2 execution backend for a forward call.

    ``"auto"`` picks a sensible default: Triton/FLA on CUDA (and the fused
    recurrent path for short inference), pure torch elsewhere (chunk for
    training/long sequences, recurrent for single-token/short inference decode).
    Forcing any other value disables automatic selection entirely.
    """
    if requested not in GDN2_BACKENDS:
        raise ValueError(
            f"Invalid GDN-2 backend '{requested}'. Valid backends: {list(GDN2_BACKENDS)}"
        )
    if requested != "auto":
        return requested

    # Safe optional Triton/FLA ops imports
    _HAS_TRITON_OPS = False
    try:
        from qwendopamine.models.gdn2.triton.chunk_gdn2 import (
            _HAS_TRITON_FLA as _CHUNK_HAS_TRITON,
        )
        from qwendopamine.models.gdn2.triton.fused_recurrent_gdn2 import (
            _HAS_TRITON_FLA as _RECURRENT_HAS_TRITON,
        )

        _HAS_TRITON_OPS = bool(_CHUNK_HAS_TRITON or _RECURRENT_HAS_TRITON)
    except (ImportError, AttributeError):
        pass

    if torch.cuda.is_available() and _HAS_TRITON_OPS:
        return "triton"
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
