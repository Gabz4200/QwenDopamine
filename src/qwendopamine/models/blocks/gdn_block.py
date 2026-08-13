from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.models.normalization import RMSNorm


class GatedDeltaNetBlock(nn.Module):
    r"""Gated DeltaNet residual block.

    Projects hidden states to a gated delta pathway, applies element-wise
    gating and additive residual update, and returns the updated hidden states.

    Args:
        config: any object with ``hidden_size`` and ``rms_norm_eps`` attributes.
        layer_idx (int): layer index for compatibility with config-driven stacks.
    """
    def __init__(self, config: Any, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.hidden_size = getattr(config, "hidden_size", 2560)
        self.norm = RMSNorm(self.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.proj = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        projected = self.proj(hidden_states)
        gated, delta, out = projected.chunk(3, dim=-1)
        hidden_states = residual + self.out_proj(out * torch.sigmoid(gated) + delta)
        return hidden_states
