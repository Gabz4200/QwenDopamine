# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Readable PyTorch reference for the GDN-2 recurrence.

Single source of truth for the GDN-2 forward and per-step backward math.
It lives beside the public op in :mod:`qwendopamine.ops` (not in the
``models/`` tree) so any caller can read the math without pulling in
the full InfiniDopamine model stack. The Taichi kernel and the
``torch_chunk_gdn2`` / ``torch_recurrent_gdn2`` production paths must
match this reference within numerical tolerance.

The recurrence (paper Eq. 10) is

.. math::

    S_t = (I - k_t (b_t \\odot k_t)^\\top) \\,\\text{Diag}(\\alpha_t) S_{t-1}
          + k_t (w_t \\odot v_t)^\\top

with the operational form used in the kernels:

.. math::

    \\bar{S}        &= \\text{Diag}(\\alpha_t)\\,S_{t-1}
    e              &= b_t \\odot k_t
    v_{\\text{ret}} &= \\bar{S}^\\top e
    v_{\\text{new}} &= (w_t \\odot v_t) - v_{\\text{ret}}
    S_t            &= \\bar{S} + k_t\\, v_{\\text{new}}^\\top
    y_t            &= S_t^\\top q_t

Shapes:

    S     : ``[B, H, K, V]``
    q_t   : ``[B, H, K]``
    k_t   : ``[B, H, K]``
    v_t   : ``[B, H, V]``
    b_t   : ``[B, H, K]``   (erase gate)
    w_t   : ``[B, H, V]``   (write gate)
    a_t   : ``[B, H, K]``   (decay = exp(g))

The per-token backward (VJP of the above) is the hand-derived one in
:func:`gdn2_reference_step_with_grad`. Both the forward and the VJP
are rank-agnostic: any state rank >= 2 works (4D ``[B, H, K, V]`` is
the production case).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDN2StepGrads:
    """Per-step gradient tuple for :func:`gdn2_reference_step_with_grad`.

    Attribute shapes mirror the inputs:

        dS : ``[B, H, K, V]``  dL/dS
        dq : ``[B, H, K]``     dL/dq_t
        dk : ``[B, H, K]``     dL/dk_t
        dv : ``[B, H, V]``     dL/dv_t
        db : ``[B, H, K]``     dL/db_t
        dw : ``[B, H, V]``     dL/dw_t
        da : ``[B, H, K]``     dL/da_t
    """

    dS: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    db: torch.Tensor
    dw: torch.Tensor
    da: torch.Tensor


def _promote(
    tensors: dict[str, torch.Tensor], state: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Promote per-token tensors to the rank of the state for einsum."""
    rank = state.dim()
    promoted: dict[str, torch.Tensor] = {}
    for key, t in tensors.items():
        if t.dim() != rank - 1:
            t = t.view(*([1] * (rank - 1 - t.dim())), *t.shape)
        promoted[key] = t
    return promoted


def _subscripts(rank: int) -> tuple[str, str, str]:
    """Build einsum subscripts from the (aligned) rank of the state."""
    lead = "".join(chr(ord("a") + i) for i in range(rank - 2))  # "bh" for 4D
    return f"{lead}kv", f"{lead}k", f"{lead}v"


def gdn2_reference_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    b_t: torch.Tensor,
    w_t: torch.Tensor,
    a_t: torch.Tensor,
    scale_qk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Single-step GDN-2 forward in the clearest form possible.

    Args:
        scale_qk: If ``True``, multiply ``q_t`` by ``K**-0.5`` so the
            reference matches the API-level scale applied by
            :func:`~qwendopamine.kernels.taichi.recurrent_taichi_gdn2`
            and :func:`~qwendopamine.models.gdn2.recurrence.recurrent.torch_recurrent_gdn2`.
            Default ``False`` keeps the reference as a pure-math form.

    Returns:
        ``(y_t, S_next)``.
    """
    if S.dim() < 2:
        raise ValueError("S must be at least 2D")
    tensors = _promote({"q": q_t, "k": k_t, "v": v_t, "b": b_t, "w": w_t}, S)
    q_t, k_t, v_t, b_t, w_t = (
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["b"],
        tensors["w"],
    )
    if scale_qk:
        q_t = q_t * (q_t.shape[-1] ** -0.5)
    # Decay the prior state column-wise. ``a_t`` is ``exp(g)``.
    S_dec = a_t.unsqueeze(-1) * S
    sub_state, sub_kv, sub_v = _subscripts(S_dec.dim())
    # Erase: the column-wise term of the rank-1 outer product that
    # subtracts from the prior state.
    k_erased = b_t * k_t  # e
    v_ret = torch.einsum(f"{sub_state},{sub_kv}->{sub_v}", S_dec, k_erased)
    v_new = w_t * v_t - v_ret
    # Rank-1 outer-product update + readout.
    S_next = S_dec + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
    y_t = torch.einsum(f"{sub_state},{sub_kv}->{sub_v}", S_next, q_t)
    return y_t, S_next


def gdn2_reference_sequence(
    S0: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    scale_qk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loop over the time axis, applying :func:`gdn2_reference_step`.

    Returns ``(y, S_T)``.
    """
    S = S0
    y_steps: list[torch.Tensor] = []
    T = q.shape[1]
    for t in range(T):
        y_t, S = gdn2_reference_step(
            S,
            q[:, t],
            k[:, t],
            v[:, t],
            b[:, t],
            w[:, t],
            a[:, t],
            scale_qk=scale_qk,
        )
        y_steps.append(y_t)
    y = torch.stack(y_steps, dim=1)
    return y, S


def gdn2_reference_step_with_grad(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    b_t: torch.Tensor,
    w_t: torch.Tensor,
    a_t: torch.Tensor,
    dy: torch.Tensor,
    scale_qk: bool = False,
) -> GDN2StepGrads:
    r"""Hand-derived single-step GDN-2 forward + per-token VJP.

    The backward path is the hand-derived VJP written out term by
    term from the operational form above, so it is independent of
    the autograd graphs of the torch and Taichi paths.

    Args:
        scale_qk: If ``True``, the same ``K**-0.5`` query scale as
            :func:`gdn2_reference_step` is applied, and ``dq`` is
            rescaled back via the chain rule.

    Returns:
        :class:`GDN2StepGrads` holding every per-input gradient.
    """
    if S.dim() < 2:
        raise ValueError("S must be at least 2D")
    tensors = _promote({"q": q_t, "k": k_t, "v": v_t, "b": b_t, "w": w_t}, S)
    q_t, k_t, v_t, b_t, w_t = (
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["b"],
        tensors["w"],
    )
    if scale_qk:
        q_t = q_t * (q_t.shape[-1] ** -0.5)
    S_dec = a_t.unsqueeze(-1) * S
    sub_state, sub_kv, sub_v = _subscripts(S_dec.dim())
    k_erased = b_t * k_t
    v_ret = torch.einsum(f"{sub_state},{sub_kv}->{sub_v}", S_dec, k_erased)
    v_new = w_t * v_t - v_ret
    S_next = S_dec + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)

    # dL/dS_next[k, d] = dy[d] * q[k]
    dS_next = q_t.unsqueeze(-1) * dy.unsqueeze(-2)  # [B,H,K,V]

    # From S_next[k, d] = S_dec[k, d] + k[k] * v_new[d]:
    #   dL/dv_new[d]       = sum_k k[k] * dS_next[k, d]
    #   dL/dk_via_write[k] = sum_d v_new[d] * dS_next[k, d]
    dv_new = (k_t.unsqueeze(-1) * dS_next).sum(dim=-2)  # [B,H,V]
    dk_write = (v_new.unsqueeze(-2) * dS_next).sum(dim=-1)  # [B,H,K]

    # From v_new[d] = w[d] * v[d] - v_ret[d]:
    #   dL/dw[d]      = dv_new[d] * v[d]
    #   dL/dv[d]      = dv_new[d] * w[d]
    #   dL/dv_ret[d]  = -dv_new[d]
    dw = dv_new * v_t
    dv = dv_new * w_t
    dv_ret = -dv_new

    # From v_ret[d] = sum_k S_dec[k, d] * (b[k] * k[k]):
    #   dL/dw_erased[k]    = sum_d S_dec[k, d] * dv_ret[d]
    #   dL/dS_dec[k, d]   += w_erased[k] * dv_ret[d]
    #   dL/db[k]           = w_erased_grad[k] * k[k]
    #   dL/dk[k]           = w_erased_grad[k] * b[k]
    dS_dec_erased = k_erased.unsqueeze(-1) * dv_ret.unsqueeze(-2)  # [B,H,K,V]
    dS_dec_from_erase = dS_next + dS_dec_erased
    dw_erased = (S_dec * dv_ret.unsqueeze(-2)).sum(dim=-1)  # [B,H,K]
    db = dw_erased * k_t
    dk_erase = dw_erased * b_t

    # From S_dec[k, d] = a[k] * S[k, d]:
    #   dL/dS[k, d]  = a[k] * dL/dS_dec[k, d]
    #   dL/da[k]     = sum_d S[k, d] * dL/dS_dec[k, d]
    dS = a_t.unsqueeze(-1) * dS_dec_from_erase
    da = (S * dS_dec_from_erase).sum(dim=-1)  # [B,H,K]

    # From y[d] = sum_k S_next[k, d] * q[k]:
    #   dL/dq[k] = sum_d S_next[k, d] * dy[d]
    dq = (S_next * dy.unsqueeze(-2)).sum(dim=-1)  # [B,H,K]
    if scale_qk:
        # dq above is dL/dq_internal where q_internal = q_t * K**-0.5.
        # Chain rule on the input scaling restores dL/dq_t.
        dq = dq * (q_t.shape[-1] ** -0.5)

    # Combine k gradients (rank-1 write + erase path).
    dk = dk_write + dk_erase

    return GDN2StepGrads(dS=dS, dq=dq, dk=dk, dv=dv, db=db, dw=dw, da=da)


__all__ = [
    "GDN2StepGrads",
    "gdn2_reference_sequence",
    "gdn2_reference_step",
    "gdn2_reference_step_with_grad",
]
