# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import GDN2Config
from .gdn2 import GatedDeltaNet2
from .rmsnorm import RMSNorm


class Block(nn.Module):
    """Transformer block with GatedDeltaNet2 token mixer."""

    def __init__(self, config: GDN2Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedDeltaNet2(config, layer_idx=layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any | None]:
        residual = x
        x = self.norm(x)
        out, _, past_key_values = self.attn(
            x, past_key_values=past_key_values, use_cache=use_cache
        )
        return residual + out, past_key_values


class GPT(nn.Module):
    """GPT architecture using GDN-2 blocks."""

    def __init__(self, config: GDN2Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Block(config, i) for i in range(1)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, past_key_values: Any | None = None
    ) -> tuple[torch.Tensor, Any | None]:
        x = self.embed(input_ids)
        for layer in self.layers:
            x, past_key_values = layer(x, past_key_values=past_key_values)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, past_key_values


__all__ = ["GPT", "Block"]
