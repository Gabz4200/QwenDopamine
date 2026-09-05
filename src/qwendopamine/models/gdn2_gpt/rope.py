"""Rotary Position Embedding utilities."""

from __future__ import annotations

import torch
from torch import Tensor

RoPECache = tuple[Tensor, Tensor]
KVCache = tuple[Tensor, Tensor]


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: float = 10000.0,
    condense_ratio: float = 1.0,
) -> RoPECache:
    r"""build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: float = 10000.0, condense_ratio: float = 1.0) -> RoPECache

    Build Rotary Position Embedding cos and sin tables.

    Args:
        seq_len (int): Maximum sequence length.
        n_elem (int): Number of embedding elements (must be even).
        dtype (torch.dtype): Output tensor dtype.
        device (torch.device): Output tensor device.
        base (float): RoPE base frequency. Default: ``10000.0``.
        condense_ratio (float): Sequence length compression factor.
            Default: ``1.0``.

    Returns:
        RoPECache: Tuple ``(cos, sin)`` of tensors ``[seq_len, n_elem]``.
    """
    theta = 1.0 / (
        base
        ** (torch.arange(0, n_elem, 2, device=device, dtype=torch.float32) / n_elem)
    )
    seq_idx = torch.arange(seq_len, device=device, dtype=torch.float32) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)
    cos = torch.cos(idx_theta).to(dtype=dtype)
    sin = torch.sin(idx_theta).to(dtype=dtype)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    r"""apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor

    Apply rotary position embedding to input tensor with exact dtype
    preservation.

    Args:
        x (torch.Tensor): Input ``[..., n_elem]``.
        cos (torch.Tensor): Cosine cache ``[seq_len, n_elem]``.
        sin (torch.Tensor): Sine cache ``[seq_len, n_elem]``.

    Returns:
        torch.Tensor: Rotated tensor of the same shape and dtype as ``x``.
    """
    orig_dtype = x.dtype
    x_f = x.float()
    rot_dim = cos.size(-1) * 2
    x_rot = x_f[..., :rot_dim]
    x_pass = x_f[..., rot_dim:]

    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    cos_expanded = cos.float().unsqueeze(0).unsqueeze(2)
    sin_expanded = sin.float().unsqueeze(0).unsqueeze(2)

    y1 = x1 * cos_expanded - x2 * sin_expanded
    y2 = x1 * sin_expanded + x2 * cos_expanded
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    if x_pass.size(-1) > 0:
        y = torch.cat((y, x_pass), dim=-1)
    return y.to(dtype=orig_dtype)
