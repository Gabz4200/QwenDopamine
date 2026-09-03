# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Hand-derived third reference for the Reinforced Delta memory core.

The RewardNet is structurally identical to Gated DeltaNet-2 with the
following channel-wise gates in place of the GDN-2 ``b``/``w``:

    S_next[d, k] = (1 - omega_E[d]) * S[d, k] + omega_W * e[d] * k[k]

where ``omega_W`` is a per-batch scalar, ``omega_E`` is a per-`d`
column-wise scalar, and ``e[d] = v[d] - sum_kk S[d, kk] * k[kk]`` is
the delta residual. The per-step state update is implemented in
:mod:`qwendopamine.models.reinforced.delta` and the per-step Taichi
adjoint in :mod:`qwendopamine.models.reinforced.taichi`.

This module provides hand-derived third references that are
**independent** of both the local torch reference and the Taichi
kernel, so they can be used to validate them. The references are
rank-agnostic (work for 2D ``[B, d, d]`` and higher-rank states).
"""

from __future__ import annotations

import torch


def _resolve_sublabels(rank: int) -> tuple[str, str, str]:
    """Build einsum subscripts from the (aligned) rank of the state."""
    lead = "".join(chr(ord("a") + i) for i in range(rank - 2))  # "" for 2D
    sub_state = f"{lead}dk"
    sub_k = f"{lead}k"
    sub_v = f"{lead}d"
    return sub_state, sub_k, sub_v


def _promote(
    tensors: dict[str, torch.Tensor],
    state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Promote per-token tensors to the rank of the state for einsum."""
    rank = state.dim()
    out: dict[str, torch.Tensor] = {}
    for k, t in tensors.items():
        if t.dim() == rank - 1:
            t = t.view(*([1] * (rank - 1 - t.dim())), *t.shape)
        out[k] = t
    return out


def canonical_delta_step(
    S: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    omega_W: torch.Tensor,
    omega_E: torch.Tensor,
) -> torch.Tensor:
    r"""Hand-derived single-step RewardNet forward.

    Shapes (per-step):

        S       : ``[B, d, d]``
        k_t     : ``[B, d]``
        v_t     : ``[B, d]``
        omega_W : ``[B]`` or ``[B, 1]``  (per-batch scalar)
        omega_E : ``[B, d]``            (per-`d` column-wise scalar)

    Returns:
        S_next : ``[B, d, d]``
    """
    rank = S.dim()
    if rank < 2:
        raise ValueError("S must be at least 2D")
    tensors = _promote({"k": k_t, "v": v_t, "omega_W": omega_W, "omega_E": omega_E}, S)
    k = tensors["k"]
    v = tensors["v"]
    ow = tensors["omega_W"]
    oe = tensors["omega_E"]
    sub_state, sub_k, sub_v = _resolve_sublabels(rank)
    # Read: e[d] = v[d] - sum_kk S[d, kk] * k[kk]
    e = torch.einsum(f"{sub_state},{sub_k}->{sub_v}", S, k).neg().add(v)
    # Rank-1 outer product + column-wise decay:
    # S_next[d, k] = (1 - omega_E[d]) * S[d, k] + omega_W * e[d] * k[k]
    outer = torch.einsum(f"{sub_v},{sub_k}->{sub_state}", e, k)
    decay = (1.0 - oe).unsqueeze(-1)  # [B, d, 1] broadcast over k
    scale = ow.unsqueeze(-1)  # [B, 1, 1]
    return decay * S + scale * outer


def canonical_delta_step_with_grad(
    S: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    omega_W: torch.Tensor,
    omega_E: torch.Tensor,
    dS_next: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Hand-derived single-step RewardNet forward + per-token VJP.

    Shapes (per-step):

        S       : ``[B, d, d]``
        k_t     : ``[B, d]``
        v_t     : ``[B, d]``
        omega_W : ``[B]`` or ``[B, 1]``
        omega_E : ``[B, d]``
        dS_next : ``[B, d, d]``  (upstream gradient w.r.t. ``S_next``)

    Returns:
        S_next   : ``[B, d, d]``
        dS       : ``[B, d, d]``  (gradient w.r.t. ``S``)
        dk       : ``[B, d]``
        dv       : ``[B, d]``
        d_omega_W: ``[B]`` or ``[B, 1]`` (matches the input rank)
        d_omega_E: ``[B, d]``
    """
    # ---- forward (operational form) ----
    rank = S.dim()
    tensors = _promote(
        {
            "k": k_t,
            "v": v_t,
            "omega_W": omega_W,
            "omega_E": omega_E,
            "dS_next": dS_next,
        },
        S,
    )
    k = tensors["k"]
    v = tensors["v"]
    ow = tensors["omega_W"]
    oe = tensors["omega_E"]
    dS = tensors["dS_next"]
    sub_state, sub_k, sub_v = _resolve_sublabels(rank)
    e = torch.einsum(f"{sub_state},{sub_k}->{sub_v}", S, k).neg().add(v)
    outer = torch.einsum(f"{sub_v},{sub_k}->{sub_state}", e, k)
    decay = (1.0 - oe).unsqueeze(-1)
    scale = ow.unsqueeze(-1)
    S_next = decay * S + scale * outer

    # ---- per-token VJP (hand-derived) ----
    # S_next[d, k] = (1 - omega_E[d]) * S[d, k] + omega_W[d] * e[d] * k[k]
    #
    # Define r[d] = sum_kk G[d, kk] * k[kk]  (upstream-weighted read).
    # Then:
    #   d_v[d]        = omega_W[d] * r[d]
    #   d_k_write[k]  = sum_dd G[dd, k] * omega_W[dd] * e[dd]
    #   d_k_read[k]   = - sum_dd S[dd, k] * omega_W[dd] * r[dd]
    #   d_k[k]        = d_k_write[k] + d_k_read[k]
    #   d_omega_W[d]  = r[d] * e[d]
    #   d_omega_E[d]  = - sum_kk G[d, kk] * S[d, kk]
    #   d_S[d, k]     = (1 - omega_E[d]) * G[d, k]
    #                   - omega_W[d] * k[k] * r[d]
    #
    # All per-channel ``omega_W[d]`` is used **inside** the sum over
    # ``d`` to keep the reduction well-defined.
    r = torch.einsum(f"{sub_state},{sub_k}->{sub_v}", dS, k)  # [B, d]
    # d_v[d] = omega_W[d] * r[d]
    dv = ow * r  # [B, d]
    # d_k_write[k] = sum_dd G[dd, k] * omega_W[dd] * e[dd]
    d_k_write = torch.einsum(f"{sub_v},{sub_state},{sub_v}->{sub_k}", ow, dS, e)
    # d_k_read[k] = - sum_dd S[dd, k] * omega_W[dd] * r[dd]
    d_k_read = -torch.einsum(f"{sub_v},{sub_v},{sub_state}->{sub_k}", ow, r, S)
    dk = d_k_write + d_k_read  # [B, k]
    # d_omega_W[d] = r[d] * e[d]
    d_omega_W_eff = r * e  # [B, d]  per-channel effective gate grad
    # d_omega_E[d] = - sum_kk G[d, kk] * S[d, kk]
    d_omega_E_eff = torch.einsum(
        f"{sub_state},{sub_state}->{sub_v}", dS, S
    ).neg()  # [B, d]
    # d_S[d, k] = (1 - omega_E[d]) * G[d, k] - omega_W[d] * k[k] * r[d]
    d_S = decay * dS - ow.unsqueeze(-1) * torch.einsum(
        f"{sub_k},{sub_v}->{sub_state}", k, r
    )
    return S_next, d_S, dk, dv, d_omega_W_eff, d_omega_E_eff


def canonical_delta_sequence(
    S0: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_W: torch.Tensor,
    omega_E: torch.Tensor,
) -> torch.Tensor:
    r"""Hand-derived sequential RewardNet forward.

    Operates on tensors of shape ``[B, T, d]`` for k/v/omega_W/omega_E.
    The state is ``[B, d, d]``.

    Returns:
        S_final : ``[B, d, d]``
    """
    T = k.shape[1]
    S = S0.clone()
    for t in range(T):
        S = canonical_delta_step(
            S=S,
            k_t=k[:, t, :],
            v_t=v[:, t, :],
            omega_W=omega_W[:, t] if omega_W.dim() > 1 else omega_W[:, t : t + 1],
            omega_E=omega_E[:, t, :],
        )
    return S


__all__ = [
    "canonical_delta_sequence",
    "canonical_delta_step",
    "canonical_delta_step_with_grad",
]
