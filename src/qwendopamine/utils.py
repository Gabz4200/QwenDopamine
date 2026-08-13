r"""Shared utilities."""

from __future__ import annotations

import torch


def get_model_device(model: torch.nn.Module) -> torch.device:
    r"""Return the device of the first model parameter.

    Falls back to ``torch.device("cpu")`` for models with no parameters.

    Args:
        model (torch.nn.Module): model whose device is queried.

    Returns:
        torch.device: device of the first parameter, or CPU.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
