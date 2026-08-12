from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.models.normalization import RMSNorm


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = getattr(config, "hidden_size", 2560)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.self_attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.mlp = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
