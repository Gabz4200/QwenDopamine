# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

"""Plasticity-aware advantage gate for the reinforced delta layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["AdvantageGate"]


class AdvantageGate(nn.Module):
    r"""Plasticity-aware advantage gate for the reinforced delta layer.

    The gate separates the global modulation into three independent
    components so that negative advantage can actively erase (instead of
    merely freezing) and large-magnitude advantage increases plasticity:

        plasticity_t = σ(W_p |A_t| + b_p)          # in (0, 1) — gates the entire update
        write_t      = σ(W_w  A_t  + b_w)          # in (0, 1) — gates the write term
        erase_t      = σ(W_e (-A_t) + b_e)          # in (0, 1) — gates the erase term

    The downstream ``DeltaMemoryCore`` consumes the triple and produces
    ``S_{t+1} = (1 - plasticity·erase·E) ⊙ S_t + (plasticity·write·W) ⊙ (e k^T)``,
    i.e. positive advantage tends to write, negative tends to erase, and the
    magnitude of advantage scales plasticity multiplicatively.

    Args:
        k_stats (int): Dimension of advantage vector A_t.
        dropout (float): Dropout probability on advantage features.

    Shape:
        - A_t: (B, k_stats)
        - Returns: (``plasticity``, ``write``, ``erase``) each (B, 1).
    """

    def __init__(
        self,
        k_stats: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if k_stats <= 0:
            raise ValueError("k_stats must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.k_stats = k_stats
        self.dropout = dropout
        self.plasticity_proj = nn.Linear(k_stats, 1)
        self.write_proj = nn.Linear(k_stats, 1)
        self.erase_proj = nn.Linear(k_stats, 1)

        with torch.no_grad():
            self.plasticity_proj.bias.zero_()
            self.plasticity_proj.weight.zero_()
            self.write_proj.bias.zero_()
            self.write_proj.weight.zero_()
            self.erase_proj.bias.zero_()
            self.erase_proj.weight.zero_()

    def forward(self, A_t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""Compute plasticity, write, and erase gates.

        Args:
            A_t: (B, k_stats) advantage vector.

        Returns:
            ``(plasticity, write, erase)`` — three (B, 1) tensors in (0, 1).
        """
        if A_t.dim() != 2:
            raise ValueError(f"Expected A_t shape (B, k_stats), got {A_t.shape}.")
        if A_t.size(-1) != self.k_stats:
            raise ValueError(f"Expected k_stats={self.k_stats}, got {A_t.size(-1)}.")

        if self.training and self.dropout > 0.0:
            A_t = F.dropout(A_t, p=self.dropout, training=True)

        abs_A = A_t.abs()
        plasticity = torch.sigmoid(self.plasticity_proj(abs_A))  # (B, 1)
        write = torch.sigmoid(self.write_proj(A_t))  # (B, 1)
        erase = torch.sigmoid(self.erase_proj(-A_t))  # (B, 1)
        return plasticity, write, erase

    def extra_repr(self) -> str:
        r"""extra_repr() -> str

        Return a string with the extra representation of the module."""
        return f"k_stats={self.k_stats}, dropout={self.dropout}"
