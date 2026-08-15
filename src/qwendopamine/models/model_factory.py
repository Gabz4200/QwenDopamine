from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.models.blocks import build_block
from qwendopamine.models.embeddings import PositionEmbeddings, TokenEmbeddings
from qwendopamine.models.normalization import RMSNorm
from qwendopamine.models.output_head import LMHead
from qwendopamine.models.qwen35 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
)


class ResearchDecoder(nn.Module):
    r"""Configurable research decoder model assembled from token/position embeddings,
    a configurable block stack, and a language-model head.

    Args:
        config: any object with model-level attributes. Supported keys:
            ``hidden_size``, ``vocab_size``, ``max_position_embeddings``,
            ``num_layers``, ``block_types``, ``hidden_dropout_prob``, ``rms_norm_eps``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = getattr(config, "hidden_size", 2560)
        self.vocab_size = getattr(config, "vocab_size", 151936)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 4096)
        self.block_types = getattr(
            config, "block_types", ["qwen"] * getattr(config, "num_layers", 4)
        )

        self.embed_tokens = TokenEmbeddings(self.vocab_size, self.hidden_size)
        self.embed_positions = PositionEmbeddings(
            self.max_position_embeddings, self.hidden_size
        )
        self.embed_dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.0))
        self.final_norm = RMSNorm(
            self.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

        self.layers = nn.ModuleList(
            [
                build_block(block_type, config, layer_idx)
                for layer_idx, block_type in enumerate(self.block_types)
            ]
        )

        self.lm_head = LMHead(self.hidden_size, self.vocab_size)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(
            position_ids
        )
        hidden_states = self.embed_dropout(hidden_states)

        for layer in self.layers:
            output = layer(hidden_states)
            hidden_states = output[0] if isinstance(output, tuple) else output

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)


def build_model(config: Any) -> nn.Module:
    r"""Build either the exact Qwen3.5 model or the research decoder from config.

    Args:
        config: model configuration instance. If an instance of :class:`Qwen3_5TextConfig`
            or :class:`Qwen3_5Config` or having ``model_type == "qwen3_5_text"``, returns
            an exact :class:`Qwen3_5ForCausalLM`. Otherwise returns :class:`ResearchDecoder`.

    Returns:
        nn.Module: assembled causal language model.
    """
    model_type = getattr(config, "model_type", None)
    if isinstance(config, (Qwen3_5TextConfig, Qwen3_5Config)) or model_type in (
        "qwen3_5_text",
        "qwen3_5",
    ):
        return Qwen3_5ForCausalLM(config)
    return ResearchDecoder(config)


def build_reference_model(
    config: Any, quantization_config: Any = None, device_map: str = "cpu", **kwargs: Any
) -> nn.Module:
    r"""Load a reference Hugging Face causal-LM model, optionally with quantization."""
    from qwendopamine.integrations.huggingface import HFIntegration

    return HFIntegration.load_model(
        model_name=getattr(config, "base_model", "Qwen/Qwen3.5-0.8B"),
        quantization_config=quantization_config,
        device_map=device_map,
        **kwargs,
    )


__all__ = [
    "Qwen3_5Config",
    "Qwen3_5ForCausalLM",
    "Qwen3_5TextConfig",
    "Qwen3_5TextModel",
    "ResearchDecoder",
    "build_model",
    "build_reference_model",
]
