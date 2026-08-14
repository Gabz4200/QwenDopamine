# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

from __future__ import annotations

import torch


def apply_rotary_emb_func(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Pure PyTorch hardware-agnostic Rotary Position Embedding (RoPE)."""
    # x: [B, T, H, D] or [B, H, T, D]
    ro_dim = cos.shape[-1] * 2
    x_ro = x[..., :ro_dim]
    x_pass = x[..., ro_dim:]

    if not interleaved:
        x1, x2 = x_ro.chunk(2, dim=-1)
        x_rot = torch.cat([-x2, x1], dim=-1)
    else:
        x1 = x_ro[..., 0::2]
        x2 = x_ro[..., 1::2]
        x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)

    cos_b = cos[..., : x1.shape[-1]]
    sin_b = sin[..., : x1.shape[-1]]
    if not interleaved:
        cos_b = torch.cat([cos_b, cos_b], dim=-1)
        sin_b = torch.cat([sin_b, sin_b], dim=-1)

    out_ro = x_ro * cos_b + x_rot * sin_b
    if x_pass.numel() > 0:
        out = torch.cat([out_ro, x_pass], dim=-1)
    else:
        out = out_ro

    return out.copy_(out) if inplace else out


__all__ = ["apply_rotary_emb_func"]
