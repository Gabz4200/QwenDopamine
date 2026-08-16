r"""Shared PyTorch device resolution and helper utilities."""

from __future__ import annotations

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


__all__ = ["get_model_device"]
