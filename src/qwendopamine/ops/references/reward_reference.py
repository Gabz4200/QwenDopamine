# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Readable PyTorch reference for the Reinforced Delta memory core.

The RewardNet is structurally identical to Gated DeltaNet-2 with the
following channel-wise gates in place of the GDN-2 ``b``/``w``:

.. math::

    e[d]        &= v[d] - \sum_{kk} S[d, kk] \cdot k[kk]
    S_{t+1}[d,k] &= (1 - \omega_E[d]) \cdot S_t[d, k]
                  + \omega_W \cdot e[d] \cdot k[k]

where ``omega_W`` is a per-batch scalar, ``omega_E`` is a per-`d`
column-wise scalar, and ``e[d]`` is the delta residual.

This module exposes the **readable, autograd-friendly** implementation
of the same math. It is independent of the Taichi kernel and of
:mod:`qwendopamine.models.reinforced.delta`. The production path
(``qwendopamine.ops.delta_core_step_out``) must match this
reference within numerical tolerance.

Shapes (per-step):

    S       : ``[B, d, d]``
    k_t     : ``[B, d]``
    v_t     : ``[B, d]``
    omega_W : ``[B]`` or ``[B, 1]``  (per-batch scalar)
    omega_E : ``[B, d]``            (per-`d` column-wise scalar)
"""

from __future__ import annotations

import torch


def reward_reference_step(
    S: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    omega_W: torch.Tensor,
    omega_E: torch.Tensor,
) -> torch.Tensor:
    """Single-step RewardNet forward in the clearest form possible.

    Returns the next state ``S_next``.
    """
    # Read the delta residual: e[d] = v[d] - sum_kk S[d, kk] * k[kk]
    e = v_t - torch.einsum("bdk,bk->bd", S, k_t)

    # Rank-1 outer product with column-wise decay.
    decay = 1.0 - omega_E  # [B, d]
    # omega_W is per-batch scalar [B]; promote to [B, 1, 1] for broadcast.
    if omega_W.dim() == 1:
        omega_W = omega_W.unsqueeze(-1)  # [B, 1]
    scale = omega_W.unsqueeze(-1)  # [B, 1, 1]
    outer = torch.einsum("bd,bk->bdk", e, k_t)
    S_next = decay.unsqueeze(-1) * S + scale * outer
    return S_next


def reward_reference_step_with_grad(
    S: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    omega_W: torch.Tensor,
    omega_E: torch.Tensor,
    dS_next: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Hand-derived per-step VJP for :func:`reward_reference_step`.

    Useful for ``torch.autograd.gradcheck``-style tests.
    """
    S_next = reward_reference_step(S, k_t, v_t, omega_W, omega_E)  # noqa: F841

    # Recompute the delta residual to keep this function stateless.
    e = v_t - torch.einsum("bdk,bk->bd", S, k_t)

    # S_next = decay * S + scale * outer
    # dS = decay * dS_next
    dS = (1.0 - omega_E).unsqueeze(-1) * dS_next
    # de = scale * (dS_next collapsed over k) -- but only the outer term
    # depends on e. S_next = decay * S + scale * (e ⊗ k) -> de = scale *
    # dS_next @ k.
    if omega_W.dim() == 1:
        omega_W_b = omega_W.unsqueeze(-1)  # [B, 1]
    else:
        omega_W_b = omega_W
    de = torch.einsum("bdk,bk->bd", dS_next, k_t) * omega_W_b
    dk_from_outer = torch.einsum("bdk,bd->bk", dS_next, e) * omega_W_b
    # dv: v_t contributes through e.  d(v - S @ k) -> dv = de
    dv = de
    # dS gets another -scale * (dS_next^T @ k ⊗ ?) for the S term in
    # e. e = v - S @ k -> dS -= scale * de ⊗ k.
    dS = dS - torch.einsum("bd,bk->bdk", de, k_t)
    # dk: k_t appears in both e and the outer term.
    dk = dk_from_outer - torch.einsum("bdk,bd->bk", S, de)
    # Column-wise decay: omega_E is a per-dim scalar, S is [B, d, k]
    # d_omega_E = -sum_k S[d, k] * dS_next[d, k]
    d_omega_E = -(S * dS_next).sum(dim=-1)
    # Per-batch scalar omega_W is per-dim outer, sum d_omega_W per-d:
    d_omega_W = (de * e).sum(dim=-1, keepdim=True)
    return {
        "dS": dS,
        "dk": dk,
        "dv": dv,
        "d_omega_W": d_omega_W,
        "d_omega_E": d_omega_E,
    }


__all__ = [
    "reward_reference_step",
    "reward_reference_step_with_grad",
]
