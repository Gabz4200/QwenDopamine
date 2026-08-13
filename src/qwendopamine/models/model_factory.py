from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.models.blocks import build_block
from qwendopamine.models.embeddings import TokenEmbeddings, PositionEmbeddings
from qwendopamine.models.normalization import RMSNorm
from qwendopamine.models.output_head import LMHead


class ResearchDecoder(nn.Module):
    r"""Configurable decoder model assembled from token/position embeddings,
    a configurable block stack, and a language-model head.

    This is the default research decoder used by :func:`build_model`.
    It mirrors standard causal-LM structure while keeping block composition
    driven by Hydra configs through :func:`qwendopamine.models.blocks.build_block`.

    Args:
        config: any object with model-level attributes. Supported keys and defaults:
            ``hidden_size`` (``2560``), ``vocab_size`` (``151936``),
            ``max_position_embeddings`` (``4096``), ``num_layers`` (``4``),
            ``block_types`` (list of block names), ``hidden_dropout_prob`` (``0.0``),
            and ``rms_norm_eps`` (``1e-6``).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = getattr(config, "hidden_size", 2560)
        self.vocab_size = getattr(config, "vocab_size", 151936)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 4096)
        self.block_types = getattr(config, "block_types", ["qwen"] * getattr(config, "num_layers", 4))

        self.embed_tokens = TokenEmbeddings(self.vocab_size, self.hidden_size)
        self.embed_positions = PositionEmbeddings(self.max_position_embeddings, self.hidden_size)
        self.embed_dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.0))
        self.final_norm = RMSNorm(self.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList([
            build_block(block_type, config, layer_idx)
            for layer_idx, block_type in enumerate(self.block_types)
        ])

        self.lm_head = LMHead(self.hidden_size, self.vocab_size)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
        hidden_states = self.embed_dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)


def build_model(config: Any) -> nn.Module:
    r"""Build the default research decoder from the provided config.

    Args:
        config: any object with model-level attributes forwarded to
            :class:`ResearchDecoder`.

    Returns:
        nn.Module: assembled :class:`ResearchDecoder`.
    """
    return ResearchDecoder(config)


def build_reference_model(config: Any, **kwargs: Any) -> nn.Module:
    r"""Load a reference Hugging Face causal-LM model, optionally with quantization.

    This helper is intended for baseline comparisons against the research decoder.

    Args:
        config: any object with ``base_model`` and HF loader kwargs.
        **kwargs: additional keyword arguments passed to
            :meth:`HFIntegration.load_model`, such as ``quantization_config``
            and ``device_map``.

    Returns:
        nn.Module: Hugging Face pretrained causal language model.
    """
    from qwendopamine.integrations.huggingface import HFIntegration
    return HFIntegration.load_model(
        model_name=config.base_model,
        quantization_config=kwargs.pop("quantization_config", None),
        device_map=kwargs.pop("device_map", "cpu"),
        **kwargs,
    )
