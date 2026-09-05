"""Build helpers for QwenDopamine HF configs and models.

Extracted from :mod:`integration` for modularity.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_infinidopamine_config(
    hidden_size: int = 2048,
    num_hidden_layers: int = 24,
    sliding_window: int = 1024,
    **kwargs: Any,
) -> Any:
    r"""build_infinidopamine_config(hidden_size: int = 2048, num_hidden_layers: int = 24, sliding_window: int = 1024, **kwargs: Any) -> Any

    Build an InfiniDopamineTextConfig instance with sensible defaults.

    Args:
        hidden_size (int): Hidden dimension. Default: ``2048``.
        num_hidden_layers (int): Number of decoder layers. Default: ``24``.
        sliding_window (int): Attention window size. Default: ``1024``.
        **kwargs: Extra config fields forwarded to
            :class:`~qwendopamine.models.infinidopamine.InfiniDopamineTextConfig`.

    Returns:
        Any: A ``InfiniDopamineTextConfig`` instance.
    """
    from qwendopamine.models.infinidopamine import InfiniDopamineTextConfig

    return InfiniDopamineTextConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        sliding_window=sliding_window,
        **kwargs,
    )


def build_infinidopamine_model(
    config: Any = None,
    **kwargs: Any,
) -> Any:
    r"""build_infinidopamine_model(config: Any = None, **kwargs: Any) -> Any

    Build an ``InfiniDopamineForCausalLM`` model instance.

    Args:
        config (Any | None): Existing config, a dict of config kwargs, or
            ``None`` to build from ``**kwargs``. Default: ``None``.
        **kwargs: Extra config fields used when ``config`` is ``None``
            or a dict.

    Returns:
        Any: An ``InfiniDopamineForCausalLM`` instance.
    """
    from qwendopamine.models.infinidopamine import (
        InfiniDopamineForCausalLM,
        InfiniDopamineTextConfig,
    )

    if config is None:
        text_cfg = InfiniDopamineTextConfig(**kwargs)
    elif isinstance(config, dict):
        text_cfg = InfiniDopamineTextConfig(**config)
    else:
        text_cfg = config
    return InfiniDopamineForCausalLM(text_cfg)


def build_gdn2_hf_config(
    hidden_size: int = 2048,
    num_hidden_layers: int = 24,
    **kwargs: Any,
) -> Any:
    r"""build_gdn2_hf_config(hidden_size: int = 2048, num_hidden_layers: int = 24, **kwargs: Any) -> Any

    Build a GDN2HFConfig instance with sensible defaults.

    Args:
        hidden_size (int): Hidden dimension. Default: ``2048``.
        num_hidden_layers (int): Number of decoder layers. Default: ``24``.
        **kwargs: Extra config fields forwarded to
            :class:`~qwendopamine.integrations.huggingface.configs.GDN2HFConfig`.

    Returns:
        Any: A ``GDN2HFConfig`` instance.
    """
    from qwendopamine.integrations.huggingface.configs import GDN2HFConfig

    return GDN2HFConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        **kwargs,
    )


def build_gdn2_hf_block(
    config: Any,
    layer_idx: int,
) -> Any:
    r"""build_gdn2_hf_block(config: Any, layer_idx: int) -> Any

    Build a single :class:`GDN2HFBlock` for the given config and layer index.

    Args:
        config (Any): The :class:`GDN2HFConfig` instance.
        layer_idx (int): Layer index.

    Returns:
        Any: A ``GDN2HFBlock`` instance.
    """
    from qwendopamine.integrations.huggingface.block import GDN2HFBlock

    return GDN2HFBlock(config=config, layer_idx=layer_idx)


def prepare_model_for_trl_training(
    model: Any,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
) -> Any:
    r"""prepare_model_for_trl_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None) -> Any

    Prepare an InfiniDopamine or Qwen model for TRL training
    (SFTTrainer, DPOTrainer, GRPOTrainer).

    Enables gradient checkpointing, ensures input embeddings calculate
    gradients, and disables the KV cache.

    Args:
        model (Any): Model to prepare.
        use_gradient_checkpointing (bool): Enable gradient checkpointing.
            Default: ``True``.
        gradient_checkpointing_kwargs (dict[str, Any] | None): Extra
            kwargs for ``gradient_checkpointing_enable``. Default: ``None``.

    Returns:
        Any: The same model, modified in-place.
    """
    if use_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=kwargs
                )
            except TypeError:
                model.gradient_checkpointing_enable()

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def _make_inputs_require_grad(
                module: nn.Module, input_t: torch.Tensor, output: torch.Tensor
            ) -> None:
                output.requires_grad_(True)

            if hasattr(model, "get_input_embeddings"):
                emb = model.get_input_embeddings()
                if emb is not None:
                    emb.register_forward_hook(_make_inputs_require_grad)

    if hasattr(model, "config") and model.config is not None:
        model.config.use_cache = False

    return model


__all__ = [
    "build_gdn2_hf_block",
    "build_gdn2_hf_config",
    "build_infinidopamine_config",
    "build_infinidopamine_model",
    "prepare_model_for_trl_training",
]
