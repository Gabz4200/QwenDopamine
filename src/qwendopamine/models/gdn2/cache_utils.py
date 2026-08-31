# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Padding-mask utilities for packed-sequence GDN-2 execution.

These helpers mirror the ``transformers`` unpadding utilities used by the
NVlabs reference: tokens with ``attention_mask == 0`` are dropped, the packed
sequence runs through the layer, and the output is scattered back to the padded
layout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Derive packed-token indices from a padding mask.

    ``1`` marks a real token, ``0`` marks padding. Returns
    ``(indices, cu_seqlens, max_seqlen_in_batch)`` where ``indices`` select the
    real tokens in the flattened ``(batch, seq)`` layout.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    r"""Gather ``x`` along the flattened ``(batch, seq)`` axis (packed layout)."""
    return x.view(-1, *x.shape[2:])[indices]


def pad_input(
    hidden: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int
) -> torch.Tensor:
    r"""Scatter a packed sequence back into the padded ``[batch, seqlen, ...]`` layout."""
    output = torch.zeros(
        batch * seqlen, *hidden.shape[1:], device=hidden.device, dtype=hidden.dtype
    )
    output[indices] = hidden
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


__all__ = ["get_unpad_data", "index_first_axis", "pad_input"]
