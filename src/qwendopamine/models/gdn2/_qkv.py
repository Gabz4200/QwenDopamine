"""Q/K/V/gate projection helpers for GDN-2.

Extracted from :mod:`block` for modularity. These are stateless
functions parameterised by the projection layers so the wrapping
:class:`~qwendopamine.models.gdn2.block.GatedDeltaNet2` can delegate to them.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def compute_qkv(
    hidden_states: torch.Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    f_proj: nn.Sequential,
    b_proj: nn.Linear,
    w_proj: nn.Linear,
    q_conv1d: nn.Module | None,
    k_conv1d: nn.Module | None,
    v_conv1d: nn.Module | None,
    use_short_conv: bool,
    conv_state_q: torch.Tensor | None,
    conv_state_k: torch.Tensor | None,
    conv_state_v: torch.Tensor | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    head_k_dim: int,
    fp32_decay: bool,
    use_cache: bool,
    cu_seqlens: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    r"""compute_qkv(hidden_states, q_proj, k_proj, v_proj, f_proj, b_proj, w_proj, q_conv1d, k_conv1d, v_conv1d, use_short_conv, conv_state_q, conv_state_k, conv_state_v, A_log, dt_bias, head_k_dim, fp32_decay, use_cache, cu_seqlens) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]

    Compute Q, K, V projections and decay gates from hidden states.

    Args:
        hidden_states (torch.Tensor): Input ``[B, T, D]``.
        q_proj (nn.Linear): Query projection.
        k_proj (nn.Linear): Key projection.
        v_proj (nn.Linear): Value projection.
        f_proj (nn.Sequential): Decay gate pre-activation MLP.
        b_proj (nn.Linear): Erase gate projection.
        w_proj (nn.Linear): Write gate projection.
        q_conv1d (nn.Module | None): Optional short-conv for Q.
        k_conv1d (nn.Module | None): Optional short-conv for K.
        v_conv1d (nn.Module | None): Optional short-conv for V.
        use_short_conv (bool): Whether to use the short-conv pre-filter.
        conv_state_q (torch.Tensor | None): Conv cache for Q.
        conv_state_k (torch.Tensor | None): Conv cache for K.
        conv_state_v (torch.Tensor | None): Conv cache for V.
        A_log (torch.Tensor): Log-decay anchor.
        dt_bias (torch.Tensor): dt bias.
        head_k_dim (int): Per-head key dimension.
        fp32_decay (bool): Upcast decay to float32.
        use_cache (bool): Whether to return updated conv states.
        cu_seqlens (Any): Cumulative sequence lengths for variable-length input.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        ``(q, k, v, g, b, w, conv_state_q, conv_state_k, conv_state_v)``.
    """
    if use_short_conv:
        assert q_conv1d is not None and k_conv1d is not None and v_conv1d is not None
        q, conv_state_q = q_conv1d(
            x=q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache or False,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = k_conv1d(
            x=k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache or False,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = v_conv1d(
            x=v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache or False,
            cu_seqlens=cu_seqlens,
        )
    else:
        q = F.silu(q_proj(hidden_states))
        k = F.silu(k_proj(hidden_states))
        v = F.silu(v_proj(hidden_states))

    g = -A_log.float().exp().repeat_interleave(head_k_dim) * F.softplus(
        f_proj(hidden_states).float() + dt_bias
    )
    g = g.float() if fp32_decay else g.to(hidden_states.dtype)

    b = b_proj(hidden_states).sigmoid()
    w = w_proj(hidden_states).sigmoid()
    return q, k, v, g, b, w, conv_state_q, conv_state_k, conv_state_v


__all__ = ["compute_qkv"]
