"""TRL training preparation helpers for QwenDopamine HF models.

Provides ``prepare_model_for_trl_training`` that enables gradient checkpointing,
ensures input embeddings calculate gradients, and verifies generation / causal-LM
training contracts for ``SFTTrainer``, ``DPOTrainer``, ``GRPOTrainer``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def prepare_model_for_trl_training(
    model: Any,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
) -> Any:
    r"""Prepare a model for TRL training (SFTTrainer, DPOTrainer, GRPOTrainer).

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.prepare_model_for_trl_training` static method body.

    Enables gradient checkpointing, ensures input embeddings calculate
    gradients, and verifies generation / causal-LM training contracts.

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
