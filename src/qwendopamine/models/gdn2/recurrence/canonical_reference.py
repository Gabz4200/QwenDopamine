# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Hand-derived third reference for the Gated DeltaNet-2 (GDN-2) recurrence.

This module is the single source of truth for GDN-2 forward and per-step
backward math. It is **independent** of both
:mod:`qwendopamine.models.gdn2.recurrence.recurrent` (local torch
recurrent reference) and :mod:`qwendopamine.kernels.taichi`
(Taichi kernels) so it can be used to validate them.

The recurrence (paper Eq. 10) is

.. math::

    S_t = (I - k_t (b_t \odot k_t)^\top) \,\text{Diag}(\alpha_t) S_{t-1}
          + k_t (w_t \odot v_t)^\top

with the operational form used in the kernels:

.. math::

    \bar{S}        &= \text{Diag}(\alpha_t)\,S_{t-1}
    e              &= b_t \odot k_t
    v_{\text{ret}} &= \bar{S}^\top e
    v_{\text{new}} &= (w_t \odot v_t) - v_{\text{ret}}
    S_t            &= \bar{S} + k_t\, v_{\text{new}}^\top
    y_t            &= S_t^\top q_t

The per-token backward (VJP of the above) is the hand-derived one in
:func:`canonical_gdn2_step_with_grad`. We compare it against both the
local torch recurrent reference and the Taichi kernel to establish
which one is wrong when they disagree.
"""

from __future__ import annotations

import torch


def canonical_gdn2_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    b_t: torch.Tensor,
    w_t: torch.Tensor,
    a_t: torch.Tensor,
    scale_qk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Hand-derived single-step GDN-2 forward.

    Shapes:
        S   : ``[B, H, K, V]``
        q_t : ``[B, H, K]``
        k_t : ``[B, H, K]``
        v_t : ``[B, H, V]``
        b_t : ``[B, H, K]``
        w_t : ``[B, H, V]``
        a_t : ``[B, H, K]``  (the already-decayed ``exp(g)`` factor)

    Args:
        scale_qk: If ``True``, multiply ``q_t`` by ``K**-0.5`` so the
            canonical matches the API-level scale applied by
            :func:`qwendopamine.kernels.taichi.recurrent_taichi_gdn2`
            and :func:`torch_recurrent_gdn2`. Default ``False`` keeps
            the canonical as a pure-math reference.

    Returns:
        y_t     : ``[B, H, V]``
        S_next  : ``[B, H, K, V]``
    """
    # Decay
    S_dec = a_t.unsqueeze(-1) * S
    if scale_qk:
        q_t = q_t * (q_t.shape[-1] ** -0.5)
    # Promote all per-token tensors to the rank of the state so the
    # canonical step is rank-agnostic. ``S_dec`` has the rank of
    # ``S`` (broadcast handles the difference in K dim).
    rank = S_dec.dim()
    if rank < 2:
        raise ValueError("S must be at least 2D")
    if q_t.dim() != rank - 1:
        # q_t lacks the trailing ``v`` axis; prepend singleton dims
        # so it broadcasts to ``(..., K)``.
        q_t = q_t.view(*([1] * (rank - 1 - q_t.dim())), *q_t.shape)
    if k_t.dim() != rank - 1:
        k_t = k_t.view(*([1] * (rank - 1 - k_t.dim())), *k_t.shape)
    if v_t.dim() != rank - 1:
        v_t = v_t.view(*([1] * (rank - 1 - v_t.dim())), *v_t.shape)
    if b_t.dim() != rank - 1:
        b_t = b_t.view(*([1] * (rank - 1 - b_t.dim())), *b_t.shape)
    if w_t.dim() != rank - 1:
        w_t = w_t.view(*([1] * (rank - 1 - w_t.dim())), *w_t.shape)
    # Build the einsum subscripts based on the (now aligned) rank.
    lead = "".join(chr(ord("a") + i) for i in range(rank - 2))  # "bh" for 4D, "" for 2D
    sub_state = f"{lead}kv"
    sub_kv = f"{lead}k"
    sub_v = f"{lead}v"
    k_erased = b_t * k_t
    v_ret = torch.einsum(f"{sub_state},{sub_kv}->{sub_v}", S_dec, k_erased)
    v_new = w_t * v_t - v_ret
    S_next = S_dec + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
    y_t = torch.einsum(f"{sub_state},{sub_kv}->{sub_v}", S_next, q_t)
    return y_t, S_next


def canonical_gdn2_step_with_grad(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    b_t: torch.Tensor,
    w_t: torch.Tensor,
    a_t: torch.Tensor,
    dy: torch.Tensor,
    scale_qk: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Hand-derived single-step GDN-2 forward + per-token VJP.

    All inputs and outputs are float32 ``[B, H, ...]``-shaped tensors
    except the per-input gradients which mirror the input shapes.

    The forward path is identical to :func:`canonical_gdn2_step`; the
    backward path is the **hand-derived** VJP, written out term by
    term from the operational form above.

    Returns:
        y_t      : ``[B, H, V]``          output
        S_next   : ``[B, H, K, V]``       new state
        dS_prev  : ``[B, H, K, V]``       ``d L / d S``
        dq_t     : ``[B, H, K]``          ``d L / d q_t``
        dk_t     : ``[B, H, K]``          ``d L / d k_t``
        dv_t     : ``[B, H, V]``          ``d L / d v_t``
        db_t     : ``[B, H, K]``          ``d L / d b_t``
        dw_t     : ``[B, H, V]``          ``d L / d w_t``
        da_t     : ``[B, H, K]``          ``d L / d a_t``
    """
    # ---- forward (operational form) ----
    S_dec = a_t.unsqueeze(-1) * S
    if scale_qk:
        q_t = q_t * (q_t.shape[-1] ** -0.5)
    rank = S_dec.dim()
    if q_t.dim() != rank - 1:
        q_t = q_t.view(*([1] * (rank - 1 - q_t.dim())), *q_t.shape)
    if k_t.dim() != rank - 1:
        k_t = k_t.view(*([1] * (rank - 1 - k_t.dim())), *k_t.shape)
    if v_t.dim() != rank - 1:
        v_t = v_t.view(*([1] * (rank - 1 - v_t.dim())), *v_t.shape)
    if b_t.dim() != rank - 1:
        b_t = b_t.view(*([1] * (rank - 1 - b_t.dim())), *b_t.shape)
    if w_t.dim() != rank - 1:
        w_t = w_t.view(*([1] * (rank - 1 - w_t.dim())), *w_t.shape)
    lead = "".join(chr(ord("a") + i) for i in range(rank - 2))
    sub_state = f"{lead}kv"
    sub_kv = f"{lead}k"
    sub_v = f"{lead}v"
    k_erased = b_t * k_t
    v_ret = torch.einsum(f"{sub_state},{sub_kv}->{sub_v}", S_dec, k_erased)
    v_new = w_t * v_t - v_ret
    S_next = S_dec + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
    y_t = torch.einsum(f"{sub_state},{sub_kv}->{sub_v}", S_next, q_t)

    # ---- per-token VJP (hand-derived from the operational form) ----
    # Derivation:
    #   y[d]   = sum_k S_next[k, d] * q[k]
    #   v_new[d] = w[d] * v[d] - v_ret[d]
    #   v_ret[d] = sum_k S_dec[k, d] * (b[k] * k[k])
    #   S_dec[k, d] = a[k] * S[k, d]
    #   S_next[k, d] = S_dec[k, d] + k[k] * v_new[d]
    #
    # Let G = d y / d S_next -> G[k, d] = q[k] (then d L / d S_next
    # is the outer product of dy with the upstream w.r.t. y).
    # Use chain rule on the per-token shape.

    # dL/dS_next[k, d] = dy[d] * q[k]
    dS_next = q_t.unsqueeze(-1) * dy.unsqueeze(-2)  # [B,H,K,V]

    # From S_next[k, d] = S_dec[k, d] + k[k] * v_new[d]
    #   dL/dS_dec[k, d]      = dS_next[k, d]
    #   dL/dv_new[d]         = sum_k k[k] * dS_next[k, d]
    #   dL/dk_via_write[k]   = sum_d v_new[d] * dS_next[k, d]
    dv_new = (k_t.unsqueeze(-1) * dS_next).sum(dim=-2)  # [B,H,V]
    dk_write = (v_new.unsqueeze(-2) * dS_next).sum(dim=-1)  # [B,H,K]

    # From v_new[d] = w[d] * v[d] - v_ret[d]
    #   dL/dw[d] = dv_new[d] * v[d]
    #   dL/dv[d] = dv_new[d] * w[d]
    #   dL/dv_ret[d] = -dv_new[d]
    dw_t = dv_new * v_t
    dv_t_from_vnew = dv_new * w_t
    dv_ret = -dv_new

    # From v_ret[d] = sum_k S_dec[k, d] * (b[k] * k[k])
    # Let w_erased[k] = b[k] * k[k]
    #   dL/dw_erased[k] = sum_d S_dec[k, d] * dv_ret[d]
    #   dL/dS_dec[k, d] += w_erased[k] * dv_ret[d]
    #   dL/db[k] = w_erased_grad[k] * k[k]
    #   dL/dk[k] = w_erased_grad[k] * b[k]
    dS_dec_erased = k_erased.unsqueeze(-1) * dv_ret.unsqueeze(-2)  # [B,H,K,V]
    dS_dec_from_erase = dS_next + dS_dec_erased
    dw_erased = (S_dec * dv_ret.unsqueeze(-2)).sum(dim=-1)  # [B,H,K]
    db_t = dw_erased * k_t
    dk_erase = dw_erased * b_t

    # From S_dec[k, d] = a[k] * S[k, d]
    #   dL/dS[k, d]      = a[k] * dL/dS_dec[k, d]
    #   dL/da[k]         = sum_d S[k, d] * dL/dS_dec[k, d]
    dS_prev = a_t.unsqueeze(-1) * dS_dec_from_erase
    da_t = (S * dS_dec_from_erase).sum(dim=-1)  # [B,H,K]

    # From y[d] = sum_k S_next[k, d] * q[k]
    # dL/dq[k] = sum_d S_next[k, d] * dy[d]
    dq_t = (S_next * dy.unsqueeze(-2)).sum(dim=-1)  # [B,H,K]
    if scale_qk:
        # dq_t above is dL/dq_internal where q_internal = q_t * K**-0.5.
        # Convert to dL/dq_t via the chain rule on the input scaling.
        dq_t = dq_t * (q_t.shape[-1] ** -0.5)

    # Combine k gradients (from rank-1 write + erase path)
    dk_t = dk_write + dk_erase

    # Add the contribution to dL/dv from the v_ret side
    dv_t = dv_t_from_vnew

    return y_t, S_next, dS_prev, dq_t, dk_t, dv_t, db_t, dw_t, da_t


def canonical_gdn2_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    scale_qk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Hand-derived sequential GDN-2 forward.

    Operates on tensors of shape ``[B, T, H, K]`` / ``[B, T, H, V]``.
    No L2 normalisation and no ``d_k**-0.5`` scaling is applied here;
    callers are responsible for the same preprocessing they apply to
    the local torch / Taichi paths so the comparison is fair.

    Args:
        scale_qk: If ``True``, multiply each ``q[:, t]`` by ``K**-0.5``
            so the canonical matches the API-level scale applied by
            :func:`recurrent_taichi_gdn2` and :func:`torch_recurrent_gdn2`.

    Returns:
        y      : ``[B, T, H, V]``
        S_last : ``[B, H, K, V]``
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if initial_state is None:
        S = torch.zeros(B, H, K, V, dtype=q.dtype, device=q.device)
    else:
        S = initial_state.to(dtype=q.dtype, device=q.device).clone()
    alpha = torch.exp(g)
    outs = []
    for t in range(T):
        y_t, S = canonical_gdn2_step(
            S=S,
            q_t=q[:, t],
            k_t=k[:, t],
            v_t=v[:, t],
            b_t=b[:, t],
            w_t=w[:, t],
            a_t=alpha[:, t],
            scale_qk=scale_qk,
        )
        outs.append(y_t)
    y = torch.stack(outs, dim=1)
    return y, S


__all__ = [
    "canonical_gdn2_sequence",
    "canonical_gdn2_step",
    "canonical_gdn2_step_with_grad",
]
