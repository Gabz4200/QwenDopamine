# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Readable PyTorch reference for the GDN-2 recurrence.

This is the **public, model-facing** reference implementation. It lives
beside the public op in :mod:`qwendopamine.ops` (not inside the
``models/`` tree) so any caller can read the math without pulling in
the full InfiniDopamine model stack. The Taichi kernel and the
``torch_chunk_gdn2`` / ``torch_recurrent_gdn2`` production paths must
match this reference within numerical tolerance.

Shapes:

    S     : ``[B, H, K, V]``
    q_t   : ``[B, H, K]``
    k_t   : ``[B, H, K]``
    v_t   : ``[B, H, V]``
    b_t   : ``[B, H, K]``   (erase gate)
    w_t   : ``[B, H, V]``   (write gate)
    a_t   : ``[B, H, K]``   (decay = exp(g))

    The forward (paper Eq. 10) is

.. math::

    S_{t+1} = (1 - b_t \odot k_t \otimes k_t) \,\text{Diag}(\alpha_t) S_t
              + k_t (w_t \odot v_t)^\top
    y_t     = S_{t+1}^\top q_t

implemented here in two readable forms: a single-step version
(:func:`gdn2_reference_step`) and a sequence version
(:func:`gdn2_reference_sequence`) that loops over time.

Channel convention for ``b`` and ``w``
---------------------------------------
The paper treats ``b`` and ``w`` as one gate per **key/value dim** —
``b`` has shape ``[B, H, K]`` and is applied element-wise on the
``k_t`` index of the state; ``w`` has shape ``[B, H, V]`` and is
applied on the ``v_t`` index. We follow that convention here.

**Note on the upstream ``Qwen3NextGatedDeltaNet``**: the upstream
HF implementation flattens this to one gate per head
(``b`` is ``[B, H]`` shared across K, ``w`` is ``[B, H, V]`` for the
value dim). That is a **coarser** parameterisation. The Qwen3.5
fork inherits the upstream shape; the GDN-2 reference (this module)
and the GatedDeltaNet2 module use the per-channel paper convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDN2StepGrads:
    """Per-step gradient tuple for :func:`gdn2_reference_step_with_grad`.

    Using a frozen dataclass instead of a dict catches typo'd key
    access at type-check time and keeps the field shape contract
    visible at the call site.

    Attributes:
        dS: ``[B, H, K, V]`` gradient with respect to the prior state.
        dk: ``[B, H, K]`` gradient with respect to the key.
        dv: ``[B, H, V]`` gradient with respect to the value.
        db: ``[B, H, K]`` gradient with respect to the erase gate.
    """

    dS: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    db: torch.Tensor


def _maybe_l2norm(x: torch.Tensor) -> torch.Tensor:
    """Apply L2 normalisation per head if requested. No-op here; the
    reference keeps the math pure (no qk L2). Production paths add it
    before the call."""
    return x


def gdn2_reference_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    b_t: torch.Tensor,
    w_t: torch.Tensor,
    a_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step GDN-2 forward in the clearest form possible.

    Returns ``(y_t, S_next)``.
    """
    # Decay the prior state column-wise. ``a_t`` is ``exp(g)``; we treat
    # it as a column-wise scalar.
    S_dec = a_t.unsqueeze(-1) * S  # [B, H, K, V] * [B, H, V] -> broadcast

    # Erase: the column-wise term of the rank-1 outer product that
    # subtracts from the prior state.
    e = b_t * k_t  # [B, H, K]
    v_ret = torch.einsum("bhkv,bhk->bhv", S_dec, e)  # S_dec^T @ e
    v_new = (w_t * v_t) - v_ret  # [B, H, V]

    # Rank-1 outer-product update.
    S_next = S_dec + torch.einsum("bhk,bhv->bhkv", k_t, v_new)
    y_t = torch.einsum("bhkv,bhk->bhv", S_next, q_t)
    return y_t, S_next


def gdn2_reference_sequence(
    S0: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loop over the time axis, applying :func:`gdn2_reference_step`.

    Returns ``(y, S_T)``.
    """
    S = S0
    y_steps: list[torch.Tensor] = []
    T = q.shape[1]
    for t in range(T):
        y_t, S = gdn2_reference_step(
            S, q[:, t], k[:, t], v[:, t], b[:, t], w[:, t], a[:, t]
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
    dy_t: torch.Tensor,
) -> GDN2StepGrads:
    """Hand-derived per-step VJP for :func:`gdn2_reference_step`.

    Useful for ``torch.autograd.gradcheck``-style tests.

    Returns:
        GDN2StepGrads: per-step gradient dataclass.
    """
    _, S_next = gdn2_reference_step(S, q_t, k_t, v_t, b_t, w_t, a_t)

    # y = S_next^T @ q  -> dS_next += q ⊗ dy  (treat q as column)
    dS_next = torch.einsum("bhk,bhv->bhkv", q_t, dy_t)
    # dS_next_t = S_dec + k v_new^T  -> v_new = S_next - S_dec
    v_new = S_next - (a_t.unsqueeze(-1) * S)
    # Outer-product backward: d(k ⊗ v_new) = dS_next → d_k = dS_next @ v_new,
    # d_v_new = dS_next^T @ k.
    dk = torch.einsum("bhkv,bhv->bhk", dS_next, v_new)
    dv_new = torch.einsum("bhkv,bhk->bhv", dS_next, k_t)
    # v_new = w * v - S_dec^T @ e  ->  dv = w * dv_new
    dv = w_t * dv_new
    # S_dec = a ⊗ S  -> dS += a ⊗ dS_dec  (read S backward)
    dS = a_t.unsqueeze(-1) * dS_next
    # e = b * k  -> de = -S_dec^T^T @ dv_new (only the -S_dec^T @ dv_new
    # path contributes, since dS_next already accounts for the direct term)
    de = -torch.einsum("bhkv,bhv->bhk", dS_next, dv_new)
    db = de * k_t
    dk = dk + de * b_t
    # da (column-wise) is not a parameter in this op; if a is the g_t
    # tensor, callers compute d_g separately.
    return GDN2StepGrads(dS=dS, dk=dk, dv=dv, db=db)


__all__ = [
    "GDN2StepGrads",
    "gdn2_reference_sequence",
    "gdn2_reference_step",
    "gdn2_reference_step_with_grad",
]
