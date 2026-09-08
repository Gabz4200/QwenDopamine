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

# Lazy imports (review H1): the HF model families are heavy and not
# needed for callers that only build a :class:`ResearchDecoder`. They
# are imported inside the factory functions below so the top-level
# ``import qwendopamine.models`` stays cheap.


class ResearchDecoder(nn.Module):
    r"""ResearchDecoder(config: Any) -> None

    Configurable research decoder assembled from token/position embeddings,
    transformer layer blocks, RMSNorm, and LM head.

    See :class:`ConfigAdapter` for config normalization.

    Args:
        config (Any): Raw configuration object from any model family.
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
        r"""forward(input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor

        Compute logits from token ids with optional explicit position ids.

        Args:
            input_ids (torch.Tensor): Token indices ``[B, T]``.
            position_ids (torch.Tensor | None): Explicit position indices
                ``[B, T]``. When ``None``, positions ``0..T-1`` are generated
                on the model's device. Default: ``None``.

        Returns:
            torch.Tensor: Vocabulary logits ``[B, T, vocab_size]``.
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
        result: torch.Tensor = self.lm_head(hidden_states)
        return result


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

_MODEL_REGISTRY: dict[str, Any] = {}


def register_model_family(name: str, builder: Any) -> None:
    r"""register_model_family(name: str, builder: Any) -> None

    Register a model builder for dynamic model creation (OCP/DIP).

    PyTorch-standard factory pattern (timm-style ``create_model``).

    Args:
        name (str): Family name (e.g. ``"infinidopamine"``, ``"qwen35"``, ``"research"``).
        builder (Any): Callable ``(config, **kwargs) -> nn.Module`` that builds the model.

    Returns:
        None
    """
    _MODEL_REGISTRY[name] = builder


def _qwen35_builder(config: Any, **kwargs: Any) -> nn.Module:
    from qwendopamine.models.qwen35 import Qwen3_5ForCausalLM

    return Qwen3_5ForCausalLM(config, **kwargs)


def _infinidopamine_builder(config: Any, **kwargs: Any) -> nn.Module:
    from qwendopamine.models.infinidopamine.model_impl import InfiniDopamineForCausalLM

    return InfiniDopamineForCausalLM(config, **kwargs)


def _research_builder(config: Any, **kwargs: Any) -> nn.Module:
    return ResearchDecoder(config, **kwargs)


register_model_family("qwen35", _qwen35_builder)
register_model_family("infinidopamine", _infinidopamine_builder)
register_model_family("research", _research_builder)


def _resolve_model_family(config: Any) -> tuple[str, Any]:
    r"""Resolve a config to a model family name and unwrapped config.

    Returns:
        tuple[str, Any]: ``(family, config)`` where ``family`` is one of
        ``"infinidopamine"``, ``"qwen35"``, or ``"unknown"``.
    """
    # Review H1: lazy import the HF model families inside the function
    # so the top-level ``import qwendopamine.models`` stays cheap.
    from qwendopamine.models.infinidopamine.configs import (
        InfiniDopamineConfig,
        InfiniDopamineTextConfig,
    )
    from qwendopamine.models.qwen35 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
    )

    config = _unwrap_text_config(config)
    model_type = getattr(config, "model_type", None)
    if isinstance(config, (InfiniDopamineTextConfig, InfiniDopamineConfig)) or (
        model_type
        and model_type in _MODEL_FAMILIES
        and model_type.startswith("infinidopamine")
    ):
        return "infinidopamine", config
    if isinstance(config, (Qwen3_5TextConfig, Qwen3_5Config)) or (
        model_type and model_type in _MODEL_FAMILIES and model_type.startswith("qwen35")
    ):
        return "qwen35", config
    return "unknown", config


def create_model(name: str, config: Any, **kwargs: Any) -> nn.Module:
    r"""create_model(name: str, config: Any, **kwargs: Any) -> nn.Module

    Dynamic factory for PyTorch-standard model creation (timm-style).

    Args:
        name (str): Registered family name (e.g. ``"qwen35"``, ``"infinidopamine"``).
        config (Any): Model configuration.
        **kwargs: Additional kwargs forwarded to the builder.

    Returns:
        nn.Module: Instantiated model.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model family {name!r}. Registered: {sorted(_MODEL_REGISTRY)}"
        )
    builder = _MODEL_REGISTRY[name]
    return builder(config, **kwargs)  # type: ignore[no-any-return]


def build_model(config: Any) -> nn.Module:
    r"""build_model(config: Any) -> nn.Module

    Instantiate a Qwen3.5/InfiniDopamine causal model or ResearchDecoder from config.

    Args:
        config (Any): Raw configuration object from any model family.

    Returns:
        nn.Module: A causal language model (``InfiniDopamineForCausalLM``,
        ``Qwen3_5ForCausalLM``, or :class:`ResearchDecoder`) depending on the
        resolved model family.
    """
    # Review H1: lazy import so the top-level ``import qwendopamine.models``
    # does not eagerly pull the HF model families.
    family, config = _resolve_model_family(config)
    if family in _MODEL_REGISTRY:
        return create_model(family, config)
    return ResearchDecoder(config)


def build_reference_model(
    config: Any, quantization_config: Any = None, device_map: str = "cpu", **kwargs: Any
) -> nn.Module:
    r"""build_reference_model(config: Any, quantization_config: Any = None, device_map: str = "cpu", **kwargs: Any) -> nn.Module

    Instantiate a reference HF causal language model with optional quantization.

    Args:
        config (Any): Raw configuration object from any model family.
        quantization_config (Any): Optional quantization configuration
            passed through ``_from_config``. Default: ``None``.
        device_map (str): Target device for model placement. Default: ``"cpu"``.
        **kwargs: Additional keyword arguments forwarded to
            ``_from_config`` (e.g. ``dtype``, ``torch_dtype``).

    Returns:
        nn.Module: Reference HuggingFace causal language model on
        ``device_map``.
    """
    # Review H1: lazy import so the top-level ``import qwendopamine.models``
    # does not eagerly pull the HF model families.
    from qwendopamine.models.infinidopamine.model_impl import InfiniDopamineForCausalLM
    from qwendopamine.models.qwen35 import Qwen3_5ForCausalLM

    family, config = _resolve_model_family(config)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if family == "infinidopamine":
        model = InfiniDopamineForCausalLM._from_config(
            config, dtype=torch.bfloat16, **kwargs
        )
    else:
        model = Qwen3_5ForCausalLM._from_config(config, dtype=torch.bfloat16, **kwargs)
    return model.to(device_map)


__all__ = [
    "ResearchDecoder",
    "build_model",
    "build_reference_model",
    "create_model",
    "register_model_family",
]
