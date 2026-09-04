# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Chunkwise pure-PyTorch GDN-2 kernels.

This module implements the reference chunkwise algorithm from the GDN-2 paper
(Appendix A). It splits the sequence into chunks of size ``C``, solves the
intra-chunk interactions via a WY representation, and carries the matrix-valued
recurrent state between chunks.

The chunkwise path is the training workhorse: it reduces the ``O(T)`` token
serialism to ``O(T/C)`` chunk steps while remaining hardware-agnostic.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange


def compute_gdn2_wy_coefficients(
    kbar: torch.Tensor,
    ebar: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Solve the WY triangular system for one chunk (paper Appendix A).

    Given the decay-normalized key ``kbar = gamma^{-1} * k`` and the
    decay-normalized erase vector ``ebar = gamma * (b * k)`` of one chunk
    (shape ``[B, H, C, K]``) plus the write vector ``z = w * v`` (shape
    ``[B, H, C, V]``), return the WY auxiliary factors

        Y = (I + T)^{-1} ebar      (shape ``[B, H, C, K]``)
        U = (I + T)^{-1} z         (shape ``[B, H, C, V]``)

    where ``T = tril(ebar @ kbar^T, -1)`` is the strictly lower-triangular
    intra-chunk interaction matrix. ``I + T`` is unit lower triangular, so the
    solves are exact, stable, and hardware agnostic.
    """
    c = kbar.shape[-2]
    t_mat = torch.tril(torch.matmul(ebar, kbar.transpose(-1, -2)), diagonal=-1)
    eye = torch.eye(c, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    a_mat = eye + t_mat  # [B, H, C, C] unit lower triangular
    y = torch.linalg.solve_triangular(a_mat, ebar, upper=False, unitriangular=True)
    u = torch.linalg.solve_triangular(a_mat, z, upper=False, unitriangular=True)
    return y, u


def compute_gdn2_intra_chunk_scores(
    q: torch.Tensor,
    gamma: torch.Tensor,
    kbar: torch.Tensor,
) -> torch.Tensor:
    r"""Build the causal intra-chunk output score matrix ``Aqk``:

    (Aqk)_{r,i} = 1_{i<=r} q_r^T Diag(gamma_r / gamma_i) k_i
    """
    q_gamma = gamma * q  # [B, H, C, K] = Diag(gamma_r) q_r
    scores = torch.matmul(q_gamma, kbar.transpose(-1, -2))  # [B, H, C, C]
    c = scores.shape[-1]
    causal = torch.tril(torch.ones(c, c, device=scores.device, dtype=torch.bool))
    return scores.masked_fill(~causal, 0.0)


def torch_chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    chunk_size: int = 64,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Pure PyTorch chunkwise GDN-2 recurrence (paper Appendix A).

    Decay-normalized chunkwise/WY formulation, built on generic torch ops
    (`matmul`, `solve_triangular`) as the reference for accelerated backends:

        kbar_r = gamma_r^{-1} * k_r        ebar_r = gamma_r * (b_r * k_r)
        z_r = w_r * v_r                    T = tril(ebar @ kbar^T, -1)
        Y = (I+T)^{-1} ebar                U = (I+T)^{-1} z
        delta = U - Y @ S_start            S_next = S_start + kbar^T @ delta
        Aqk = causal_tril(gamma*q @ kbar^T)
        output = gamma*q @ S_start + Aqk @ delta
    """
    batch_size, seq_len, num_heads, d_k = q.shape
    d_v = v.shape[-1]
    out_dtype = q.dtype

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    b_f = b.float()
    w_f = w.float()

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

    scale = d_k**-0.5
    q = q * scale

    # Move to chunk-friendly layout [B, H, T, D].
    q = rearrange(q, "b t h d -> b h t d")
    k = rearrange(k, "b t h d -> b h t d")
    v = rearrange(v, "b t h d -> b h t d")
    g = rearrange(g, "b t h d -> b h t d")
    b_f = rearrange(b_f, "b t h d -> b h t d")
    w_f = rearrange(w_f, "b t h d -> b h t d")

    if initial_state is None:
        state = torch.zeros(
            batch_size, num_heads, d_k, d_v, dtype=torch.float32, device=q.device
        )
    else:
        # At position 0 the cumulative decay reaches 1, so the normalized state
        # equals the real-space state (matches the reference oracle).
        state = initial_state.float()

    outputs: list[torch.Tensor] = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        q_c = q[:, :, start:end]
        k_c = k[:, :, start:end]
        v_c = v[:, :, start:end]
        g_c = g[:, :, start:end]
        b_c = b_f[:, :, start:end]
        w_c = w_f[:, :, start:end]

        # Chunk-local cumulative decay gamma: [B, H, C, K].
        gamma = torch.exp(torch.cumsum(g_c, dim=2))
        gamma_last = gamma[:, :, -1:, :]  # [B, H, 1, K]

        # Decay-normalized factors (paper Eq. 33).
        gam_safe = gamma.clamp_min(1e-12)
        kbar = k_c / gam_safe  # [B, H, C, K]
        ebar = gamma * (b_c * k_c)  # [B, H, C, K]
        z = w_c * v_c  # [B, H, C, V]

        # WY triangular solve.
        y, u = compute_gdn2_wy_coefficients(kbar, ebar, z, device=q.device)

        # Normalized state correction: delta = U - Y @ S_start.
        delta = u - torch.matmul(y, state)  # [B, H, C, V]

        # Output read: out = (gamma*q) @ S_start + Aqk @ delta.
        q_gamma = gamma * q_c  # [B, H, C, K]
        out_inter = torch.matmul(q_gamma, state)  # [B, H, C, V]
        aqk = compute_gdn2_intra_chunk_scores(q_c, gamma, kbar)  # [B, H, C, C]
        out_c = out_inter + torch.matmul(aqk, delta)
        outputs.append(out_c)

        # Carry state to next chunk: S_next = gamma_last^T * (S_start + kbar^T @ delta).
        state = gamma_last.transpose(-1, -2) * (
            state + torch.matmul(kbar.transpose(-1, -2), delta)
        )

    out = torch.cat(outputs, dim=2)  # [B, H, T, V]
    out = rearrange(out, "b h t d -> b t h d").to(out_dtype)

    final_state: torch.Tensor | None = None
    if output_final_state:
        final_state = state.to(out_dtype)

    return out, final_state


__all__ = [
    "compute_gdn2_intra_chunk_scores",
    "compute_gdn2_wy_coefficients",
    "torch_chunk_gdn2",
]
