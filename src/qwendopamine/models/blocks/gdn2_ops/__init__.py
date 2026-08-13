r"""GDN-2 kernel dispatch.

Deferred-import wrapper around ``flash-linear-attention``'s ``chunk_gdn2``.
Falls back to the pure-PyTorch recurrence when CUDA/FLA are unavailable.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from qwendopamine.models.blocks.gdn2 import GDN2Mixer


def dispatch_gdn2(
    mixer: GDN2Mixer,
    hidden_states: Any,
    proj: Any,
    **kwargs: Any,
) -> Any:
    r"""Dispatch GDN-2 to the chunkwise GPU kernel when available, else fallback.

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
    if not torch.cuda.is_available():
        warnings.warn("CUDA unavailable; using pure-PyTorch GDN-2 recurrence.")
        return mixer._forward_fallback(
            hidden_states, proj.q, proj.k, proj.v, proj.alpha, proj.b, proj.w
        )

    module_tensor = next(mixer.parameters())
    if module_tensor.device.type != "cuda":
        return mixer._forward_fallback(
            hidden_states, proj.q, proj.k, proj.v, proj.alpha, proj.b, proj.w
        )

    try:
        from flash_linear_attention.ops.gdn2 import chunk_gdn2  # type: ignore[import]
    except ImportError:
        warnings.warn("flash-linear-attention unavailable; using pure-PyTorch GDN-2 recurrence.")
        return mixer._forward_fallback(
            hidden_states, proj.q, proj.k, proj.v, proj.alpha, proj.b, proj.w
        )

    B, T, _ = hidden_states.shape
    q = proj.q
    k = proj.k
    v = proj.v
    alpha = proj.alpha
    b = proj.b
    w = proj.w

    scale = mixer.head_dim**-0.5
    o, _ = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=alpha,
        b=b,
        wg=w,
        scale=scale,
        chunk_size=64,
        cu_seqlens=None,
    )

    o = o.transpose(1, 2).contiguous().view(B, T, mixer.value_dim)

    gate = mixer.g_proj(hidden_states)
    gate = gate.view(B, T, mixer.num_v_heads, mixer.head_v_dim).transpose(1, 2)
    o = mixer.o_norm(o, gate)
    o = o.transpose(1, 2).contiguous().view(B, T, mixer.value_dim)
    return mixer.o_proj(o)
