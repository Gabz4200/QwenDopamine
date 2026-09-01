r"""Model factory and configurable research decoder assembly for Qwen architectures."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.models.blocks import build_block
from qwendopamine.models.core.config_adapter import ConfigAdapter
from qwendopamine.models.core.embeddings import PositionEmbeddings, TokenEmbeddings
from qwendopamine.models.core.normalization import RMSNorm
from qwendopamine.models.core.output_head import LMHead
from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineForCausalLM,
    InfiniDopamineTextConfig,
)
from qwendopamine.models.qwen35 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)


class ResearchDecoder(nn.Module):
    r"""Configurable research decoder assembled from token/position embeddings, transformer layer blocks, RMSNorm, and LM head."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        config = _unwrap_text_config(config)
        self.config = ConfigAdapter(config, family="research")
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.max_position_embeddings = self.config.max_position_embeddings

        self.tok_embeddings = TokenEmbeddings(self.vocab_size, self.hidden_size)
        self.pos_embeddings = PositionEmbeddings(
            self.max_position_embeddings, self.hidden_size
        )

        block_types = getattr(self.config, "block_types", None)
        if block_types is None:
            num_layers = self.config.num_hidden_layers
            default_block = getattr(self.config, "block_type", "qwen")
            block_types = [default_block] * num_layers

        self.layers = nn.ModuleList(
            [
                build_block(bt, config, layer_idx=idx)
                for idx, bt in enumerate(block_types)
            ]
        )
        self.norm = RMSNorm(self.hidden_size)
        self.lm_head = LMHead(self.hidden_size, self.vocab_size)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Compute logits from token ids with optional explicit position ids."""
        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

        hidden_states = self.tok_embeddings(input_ids) + self.pos_embeddings(
            position_ids
        )
        for layer in self.layers:
            out = layer(hidden_states)
            hidden_states = out[0] if isinstance(out, tuple) else out
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


def _unwrap_text_config(config: Any) -> Any:
    r"""Unwrap composite configs that delegate to a ``text_config`` attribute."""
    if hasattr(config, "text_config") and not hasattr(config, "vocab_size"):
        return config.text_config
    return config


_MODEL_FAMILIES = (
    "infinidopamine",
    "infini_dopamine",
    "infinidopamine_text",
    "infinidopamine_reference",
    "qwen35",
    "qwen3_5",
    "qwen35_text",
    "qwen35_reference",
)


def _resolve_model_family(config: Any) -> tuple[str, Any]:
    r"""Resolve a config to a model family name and unwrapped config.

    Returns:
        tuple[str, Any]: ``(family, config)`` where ``family`` is one of
        ``"infinidopamine"``, ``"qwen35"``, or ``"unknown"``.
    """
    config = _unwrap_text_config(config)
    model_type = getattr(config, "model_type", None)
    if isinstance(config, (InfiniDopamineTextConfig, InfiniDopamineConfig)) or (
        model_type and model_type in _MODEL_FAMILIES and model_type.startswith("infinidopamine")
    ):
        return "infinidopamine", config
    if isinstance(config, (Qwen3_5TextConfig, Qwen3_5Config)) or (
        model_type and model_type in _MODEL_FAMILIES and model_type.startswith("qwen35")
    ):
        return "qwen35", config
    return "unknown", config


def build_model(config: Any) -> nn.Module:
    r"""Instantiate a Qwen3.5/InfiniDopamine causal model or ResearchDecoder from config."""
    family, config = _resolve_model_family(config)
    if family == "infinidopamine":
        return InfiniDopamineForCausalLM(config)
    if family == "qwen35":
        return Qwen3_5ForCausalLM(config)
    return ResearchDecoder(config)


def build_reference_model(
    config: Any, quantization_config: Any = None, device_map: str = "cpu", **kwargs: Any
) -> nn.Module:
    r"""Instantiate a reference HF causal language model with optional quantization."""
    family, config = _resolve_model_family(config)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if family == "infinidopamine":
        model = InfiniDopamineForCausalLM._from_config(
            config, dtype=torch.bfloat16, **kwargs
        )
    else:
        model = Qwen3_5ForCausalLM._from_config(
            config, dtype=torch.bfloat16, **kwargs
        )
    return model.to(device_map)


__all__ = [
    "ResearchDecoder",
    "build_model",
    "build_reference_model",
]
