r"""GDN-2 kernel dispatch.

Deferred-import wrapper around ``flash-linear-attention``'s ``chunk_gdn2``.
Falls back to the pure-PyTorch recurrence when CUDA/FLA are unavailable.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch.nn import functional as F

from qwendopamine.models.blocks.gdn2 import GDN2Mixer, GDN2Projections, _gated_delta_rule_2_fallback


# Warn only once per process; the fallback is the expected CPU path.
_warned_fallback = False


def _warn_fallback(reason: str) -> None:
    global _warned_fallback
    if not _warned_fallback:
        warnings.warn(f"GDN-2 falling back to pure-PyTorch recurrence: {reason}")
        _warned_fallback = True


def _infer_device(tensor: torch.Tensor) -> str:
    """Return the device type string from the input tensor."""
    return tensor.device.type


def dispatch_gdn2(
    mixer: GDN2Mixer,
    hidden_states: torch.Tensor,
    proj: GDN2Projections,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Dispatch GDN-2 to the chunkwise GPU kernel when available, else fallback.

    The fallback path is the pure-PyTorch token-by-token recurrence, which
    works on any device (CPU, CUDA, ROCm, MPS).  The chunkwise kernel is
    only used when the input lives on a CUDA device and
    ``flash-linear-attention`` is installed.

    Args:
        mixer: :class:`GDN2Mixer` instance whose parameters and configuration
            determine kernel eligibility.
        hidden_states: input tensor of shape ``[B, T, hidden_size]``.
        proj: :class:`GDN2Projections` container with precomputed ``q``, ``k``,
            ``v``, ``alpha``, ``b``, and ``w`` tensors.
        **kwargs: unused; present for interface compatibility.

    Returns:
        Output tensor of shape ``[B, T, hidden_size]``.
    """
    device = _infer_device(hidden_states)

    if device != "cuda":
        _warn_fallback(f"input on {device}; chunk kernel requires CUDA")
        return _gated_delta_rule_2_fallback(mixer, hidden_states, proj)

    try:
        from flash_linear_attention.ops.gdn2 import chunk_gdn2  # type: ignore[import]
    except ImportError:
        _warn_fallback("flash-linear-attention not installed")
        return _gated_delta_rule_2_fallback(mixer, hidden_states, proj)

    B, T, _ = hidden_states.shape
    q, k, v = proj.q, proj.k, proj.v
    alpha, b, w = proj.alpha, proj.b, proj.w

    o, _ = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=alpha,
        b=b,
        wg=w,
        scale=mixer.head_dim**-0.5,
        chunk_size=64,
        cu_seqlens=None,
    )

    out = o.transpose(1, 2).contiguous().view(B, T, mixer.value_dim)
    g = mixer.g_proj(hidden_states)
    out = mixer.o_norm(out) * F.silu(g)
    return mixer.o_proj(out)
