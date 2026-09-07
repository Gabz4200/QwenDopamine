"""Gate regularization and entropy for InfiniDopamine GatedDeltaNet.

Extracted from ``_gated_delta_net.py`` for size and SRP.

Standalone functions that compute gate regularization loss and entropy.
These accept a model instance as their first argument so that
``InfiniDopamineGatedDeltaNet.get_gate_regularization_loss()`` and
``InfiniDopamineGatedDeltaNet.get_gate_entropy()`` can delegate to them.
"""

from __future__ import annotations

from typing import Any

import torch


def gate_regularization_loss(
    model: Any,
    target: float = 0.5,
    hidden_states: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute mean squared deviation of data-dependent routing gates from target balance.

    Extracted from :meth:`InfiniDopamineGatedDeltaNet.get_gate_regularization_loss`
    for size and SRP.

    Args:
        model: The :class:`InfiniDopamineGatedDeltaNet` instance.
        target: Target gate value (default ``0.5``).
        hidden_states: Optional hidden states to compute gates from.

    Returns:
        Scalar tensor of mean squared deviation.
    """
    # Gate computation from original get_gate_regularization_loss
    if hidden_states is None:
        # Use stored betas; at init betas=0 so gate=sigmoid(0)=0.5
        gate = torch.sigmoid(model.betas)
    else:
        gate_logits = model.betas + model.in_proj_gate(hidden_states).unsqueeze(-1)
        gate = torch.sigmoid(gate_logits)

    # Mean squared deviation from target
    loss = torch.mean((gate - target) ** 2)
    return loss


def gate_entropy(
    model: Any,
    hidden_states: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute Shannon entropy of the routing gate distribution across heads and tokens.

    Extracted from :meth:`InfiniDopamineGatedDeltaNet.get_gate_entropy`
    for size and SRP.

    Args:
        model: The :class:`InfiniDopamineGatedDeltaNet` instance.
        hidden_states: Optional hidden states to compute gates from.

    Returns:
        Scalar tensor of mean entropy across heads and tokens.
    """
    # Compute gate from model.betas and in_proj_gate
    if hidden_states is None:
        gate = torch.sigmoid(model.betas)
    else:
        gate_logits = model.betas + model.in_proj_gate(hidden_states).unsqueeze(-1)
        gate = torch.sigmoid(gate_logits)

    # Compute entropy: -p * log(p) - (1-p) * log(1-p) for each gate value
    eps = torch.finfo(gate.dtype).tiny
    entropy = -(gate * (gate + eps).log() + (1 - gate + eps) * (1 - gate + eps).log())
    return entropy.mean()


__all__ = ["gate_entropy", "gate_regularization_loss"]