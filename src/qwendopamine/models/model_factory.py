r"""Model factory and configurable research decoder assembly for Qwen architectures."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.models.blocks import build_block
from qwendopamine.models.config_adapter import ConfigAdapter
from qwendopamine.models.embeddings import PositionEmbeddings, TokenEmbeddings
from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineForCausalLM,
    InfiniDopamineTextConfig,
)
from qwendopamine.models.normalization import RMSNorm
from qwendopamine.models.output_head import LMHead
from qwendopamine.models.qwen35 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)


class ResearchDecoder(nn.Module):
    r"""ResearchDecoder(config)

    Configurable research decoder model assembled from token/position embeddings, sequence of registered
    transformer layer blocks, final RMSNorm, and output language model prediction head.

    Args:
        config (Any): Architecture configuration instance containing hyperparameters (e.g. ``hidden_size``,
            ``vocab_size``, ``block_types``, ``num_hidden_layers``).

    Examples::

        >>> import types
        >>> cfg = types.SimpleNamespace(
        ...     hidden_size=64, vocab_size=1000, max_position_embeddings=512,
        ...     num_hidden_layers=2, block_types=["qwen", "gdn2"]
        ... )
        >>> model = ResearchDecoder(cfg)
        >>> input_ids = torch.tensor([[1, 2, 4], [5, 6, 7]])
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 3, 1000])
    """

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
        r"""forward(input_ids, position_ids=None) -> Tensor

        Args:
            input_ids (Tensor): Input token ID sequence tensor of shape :math:`(B, L)`.
            position_ids (Tensor, optional): Optional explicit position indices of shape :math:`(B, L)`.
                Default: ``None``.

        Returns:
            Tensor: Output logit sequence tensor of shape :math:`(B, L, \text{vocab\_size})`.
        """
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
    r"""build_model(config) -> nn.Module

    Instantiates either a standard Qwen3.5/InfiniDopamine causal model or custom ResearchDecoder based on config.

    Args:
        config (Any): Architecture configuration instance.

    Returns:
        nn.Module: Instantiated model instance.
    """
    family, config = _resolve_model_family(config)
    if family == "infinidopamine":
        return InfiniDopamineForCausalLM(config)
    if family == "qwen35":
        return Qwen3_5ForCausalLM(config)
    return ResearchDecoder(config)


def build_reference_model(
    config: Any, quantization_config: Any = None, device_map: str = "cpu", **kwargs: Any
) -> nn.Module:
    r"""build_reference_model(config, quantization_config=None, device_map="cpu", **kwargs) -> nn.Module

    Instantiates a reference Hugging Face causal language model with optional quantization setup.

    Args:
        config (Any): Architecture configuration instance.
        quantization_config (Any, optional): Hugging Face bitsandbytes or quantization configuration.
            Default: ``None``.
        device_map (str, optional): Target device mapping string. Default: ``"cpu"``.
        **kwargs (Any): Additional keyword arguments passed to model constructor.

    Returns:
        nn.Module: Instantiated model instance.
    """
    family, config = _resolve_model_family(config)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if family == "infinidopamine":
        return InfiniDopamineForCausalLM._from_config(
            config, device_map=device_map, torch_dtype=torch.bfloat16, **kwargs
        )
    return Qwen3_5ForCausalLM._from_config(
        config, device_map=device_map, torch_dtype=torch.bfloat16, **kwargs
    )


__all__ = [
    "ResearchDecoder",
    "build_model",
    "build_reference_model",
]
