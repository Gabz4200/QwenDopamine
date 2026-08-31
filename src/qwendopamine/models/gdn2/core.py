# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Pure PyTorch GDN-2 recurrence engine.

This module provides the hardware-agnostic reference implementation of the
Gated DeltaNet-2 matrix state update. It includes:

- :func:`gated_delta_2_step` -- single-token functional step
- :func:`process_sequence_recurrence` -- sequential loop over a time dimension
- :func:`torch_recurrent_gdn2` -- public recurrent forward kernel

All operations use standard PyTorch primitives and run on CPU or GPU without
Triton/CUDA dependencies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gated_delta_2_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    b_t: torch.Tensor,
    w_t: torch.Tensor,
    a_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Execute one step of the Gated DeltaNet-2 matrix state update.

    Equations (paper Eq. 3 / NVlabs reference):

        S_t = (I - k_t (b_t \odot k_t)^T) \text{diag}(a_t) S_{t-1} + k_t (w_t \odot v_t)^T

    Expanding the first term gives the operational form used below:

        v_{\text{retrieved}} = \text{diag}(a_t) S_{t-1} (b_t \odot k_t)
        v_{\text{write}} = w_t \odot v_t - v_{\text{retrieved}}
        S_t = \text{diag}(a_t) S_{t-1} + k_t v_{\text{write}}^\top
        y_t = S_t q_t

    Args:
        S: Recurrent memory state of shape ``[B, H, K, V]``.
        q_t: Query vector of shape ``[B, H, K]``.
        k_t: Key vector of shape ``[B, H, K]``.
        v_t: Value vector of shape ``[B, H, V]``.
        b_t: Channel-wise erase gate of shape ``[B, H, K]``.
        w_t: Channel-wise write gate of shape ``[B, H, V]``.
        a_t: Channel-wise decay factors of shape ``[B, H, K]``.

    Returns:
        y_t: Output vector of shape ``[B, H, V]``.
        S_next: Updated memory state of shape ``[B, H, K, V]``.
    """
    # Channel-wise decay is applied before the key-side retrieval, matching
    # the nvlabs recurrence: S_t = (I - k_t (b_t * k_t)^T) diag(a_t) S_{t-1} + ...
    state_decayed = a_t.unsqueeze(-1) * S  # [B, H, K, V]
    k_erased = b_t * k_t  # [B, H, K]
    v_retrieved = torch.einsum("bhkv,bhk->bhv", state_decayed, k_erased)  # [B, H, V]

    # Decoupled Delta write: (w_t * v_t) - v_retrieved
    v_write = w_t * v_t - v_retrieved  # [B, H, V]

    # Outer-product update: k_t v_write^T
    update = k_t.unsqueeze(-1) * v_write.unsqueeze(-2)  # [B, H, K, V]
    S_next = state_decayed + update

    # Query readout
    y_t = torch.einsum("bhkv,bhk->bhv", S_next, q_t)
    return y_t, S_next


def process_sequence_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the recurrent GDN-2 engine across a full time sequence.

    Args:
        q: Queries of shape ``[B, T, H, K]``.
        k: Keys of shape ``[B, T, H, K]``.
        v: Values of shape ``[B, T, H, V]``.
        b: Erase gates of shape ``[B, T, H, K]``.
        w: Write gates of shape ``[B, T, H, V]``.
        alpha: Channel-wise decay of shape ``[B, T, H, K]``.
        initial_state: Optional initial state ``[B, H, K, V]``.

    Returns:
        y: Output sequence ``[B, T, H, V]``.
        S_final: Final recurrent state ``[B, H, K, V]``.
    """
    B, T, H, K = q.shape
    if initial_state is None:
        S = torch.zeros(B, H, K, K, device=q.device, dtype=q.dtype)
    else:
        S = initial_state

    outputs = []
    for t in range(T):
        y_t, S = gated_delta_2_step(
            S=S,
            q_t=q[:, t],
            k_t=k[:, t],
            v_t=v[:, t],
            b_t=b[:, t],
            w_t=w[:, t],
            a_t=alpha[:, t],
        )
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)  # [B, T, H, V]
    return y, S


def torch_recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Pure PyTorch token-by-token GDN-2 recurrence loop (reference oracle).

    Args:
        q: Queries ``[B, T, H, K]``.
        k: Keys ``[B, T, H, K]``.
        v: Values ``[B, T, H, V]``.
        g: Log-decay gate ``[B, T, H, K]``.
        b: Channel-wise erase gate ``[B, T, H, K]``.
        w: Channel-wise write gate ``[B, T, H, V]``.
        initial_state: Optional initial state ``[B, H, K, V]``.
        output_final_state: If ``True``, return the final state.
        use_qk_l2norm_in_kernel: Apply L2 normalization to q/k.

    Returns:
        out: Output tensor ``[B, T, H, V]``.
        final_state: Final state ``[B, H, K, V]`` or ``None``.
    """
    batch_size, seq_len, num_heads, d_k = q.shape
    d_v = v.shape[-1]
    dtype = q.dtype

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

    if initial_state is None:
        state = torch.zeros(
            batch_size, num_heads, d_k, d_v, dtype=torch.float32, device=q.device
        )
    else:
        state = initial_state.float()

    outputs = []
    exp_g = torch.exp(g)

    for t in range(seq_len):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        g_t = exp_g[:, t]
        b_t = b_f[:, t]
        w_t = w_f[:, t]

        # Channel-wise decay
        state = state * g_t.unsqueeze(-1)

        # Gated DeltaNet-2 update
        k_erased = b_t * k_t
        v_retrieved = torch.einsum("bhkv,bhk->bhv", state, k_erased)
        v_write = w_t * v_t - v_retrieved
        state = state + k_t.unsqueeze(-1) * v_write.unsqueeze(-2)

        # Output read
        out_t = torch.einsum("bhkv,bhk->bhv", state, q_t)
        outputs.append(out_t)

    out = torch.stack(outputs, dim=1).to(dtype)
    final_state = state.to(dtype) if output_final_state else None

    return out, final_state


__all__ = [
    "gated_delta_2_step",
    "process_sequence_recurrence",
    "torch_recurrent_gdn2",
]
