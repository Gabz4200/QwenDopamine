r"""Shared PyTorch device resolution and helper utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def get_model_device(model: nn.Module) -> torch.device:
    r"""get_model_device(model) -> torch.device

    Returns the target execution device of the first parameter in a PyTorch module.

    Falls back to ``torch.device("cpu")`` for empty models or modules with no parameters.

    Args:
        model (nn.Module): PyTorch module instance whose device is queried.

    Returns:
        torch.device: Device of the first parameter, or CPU device if no parameters exist.

    Examples::

        >>> model = nn.Linear(10, 5)
        >>> device = get_model_device(model)
        >>> device.type
        'cpu'
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_to_device(batch: Any, device: torch.device) -> Any:
    r"""move_to_device(batch, device) -> Any

    Recursively move tensors in a batch to the target device.

    Args:
        batch (Any): Tensor, dict of tensors, or nested structure.
        device (torch.device): Target device.

    Returns:
        Any: Batch with tensors moved to the target device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    return batch


__all__ = ["get_model_device", "move_to_device"]
