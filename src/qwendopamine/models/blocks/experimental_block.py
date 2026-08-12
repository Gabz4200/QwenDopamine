from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.models.normalization import RMSNorm


class ExperimentalBlock(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = getattr(config, "hidden_size", 2560)
        self.norm = RMSNorm(self.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.scale = nn.Parameter(torch.full((self.hidden_size,), float(getattr(config, "new_block_scale", 0.001))))
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj(hidden_states) * self.scale
        return residual + hidden_states
