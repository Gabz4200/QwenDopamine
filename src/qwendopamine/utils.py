r"""Shared PyTorch device resolution and helper utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def get_model_device(model: nn.Module) -> torch.device:
    r"""Return the device of the first parameter, falling back to CPU."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_to_device(batch: Any, device: torch.device) -> Any:
    r"""Recursively move tensors in a batch to the target device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            k: move_to_device(v, device) for k, v in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        moved = [move_to_device(item, device) for item in batch]
        return type(batch)(moved)
    return batch


__all__ = ["get_model_device", "move_to_device"]
