"""Gate balance regularization loss for InfiniDopamine text tower.

Extracted from :mod:`model` for size. The loss is the mean of each
GDN-2 layer's gate regularization term.
"""

from __future__ import annotations

from typing import Any

import torch


def gate_regularization_loss(model: Any, target: float = 0.5) -> torch.Tensor:
    r"""gate_regularization_loss(model, target=0.5) -> torch.Tensor

    Sum gate balance regularization across all GDN-2 layer blocks.

    Args:
        model (Any): The :class:`InfiniDopamineTextModel` instance.
        target (float): Target gate balance value. Default: ``0.5``.

    Returns:
        torch.Tensor: Scalar regularization loss.
    """
    losses: list[torch.Tensor] = []
    for layer in model.layers[: model.config.num_hidden_layers]:
        if hasattr(layer, "linear_attn") and hasattr(
            layer.linear_attn, "get_gate_regularization_loss"
        ):
            linear_attn: Any = layer.linear_attn
            losses.append(linear_attn.get_gate_regularization_loss(target=target))
    if not losses:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def parallel_reward_gate_loss(model: Any) -> torch.Tensor:
    r"""parallel_reward_gate_loss(model) -> torch.Tensor

    Mean ``σ(W_g x + b_g) - init_bias`` across all parallel reward gates.

    Penalises the gate from drifting away from its initialisation
    (``sigmoid(init_bias) ≈ 0.0067`` by default) so the dopamine branch
    stays effectively silent until the rest of the model has stabilised.
    Active layers only.

    Args:
        model (Any): The :class:`InfiniDopamineForCausalLM` instance.

    Returns:
        torch.Tensor: Scalar penalty on gate deviation.
    """
    device = next(model.parameters()).device
    init_bias = float(getattr(model.config, "reward_gate_init_bias", -5.0))
    init_gate = float(torch.sigmoid(torch.tensor(init_bias)).item())
    # Use a fixed zero-like input so the gate ≈ sigmoid(bias) on init.
    # The exact value doesn't matter — only the deviation matters.
    losses: list[torch.Tensor] = []
    for layer in model.model.layers[: model.config.num_hidden_layers]:
        if not hasattr(layer, "reward_gate_proj"):
            continue
        gate = torch.sigmoid(layer.reward_gate_proj.bias)
        losses.append(((gate - init_gate) ** 2).mean())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


__all__ = ["gate_regularization_loss", "parallel_reward_gate_loss"]
