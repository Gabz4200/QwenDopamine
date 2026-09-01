"""Rotary position embeddings for InfiniDopamine models."""

from __future__ import annotations

from typing import Any

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
)

from qwendopamine.models.infinidopamine.configs import InfiniDopamineTextConfig


class InfiniDopamineVisionRotaryEmbedding(Qwen3VLVisionRotaryEmbedding):
    """Vision rotary embedding for InfiniDopamine."""


class InfiniDopamineTextRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    """Text rotary embedding with InfiniDopamine-specific RoPE parameters."""

    def __init__(self, config: InfiniDopamineTextConfig, device: Any = None) -> None:
        super().__init__(config)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: InfiniDopamineTextConfig, device: Any = None, **kwargs: Any
    ) -> tuple[torch.Tensor, float]:
        _ = kwargs
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor
