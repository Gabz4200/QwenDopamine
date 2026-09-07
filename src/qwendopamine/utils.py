r"""Shared PyTorch device resolution and helper utilities."""

from __future__ import annotations

from typing import Any

import torch

from qwendopamine.models.ports import ModelPort


def get_model_device(model: ModelPort) -> torch.device:
    r"""get_model_device(model: ModelPort) -> torch.device

    Return the device of the first parameter, falling back to CPU.

    Args:
        model (ModelPort): Model whose parameters are inspected.

    Returns:
        torch.device: Device of the first parameter, or ``torch.device("cpu")``
        if the model has no parameters.
    """
    try:
        return next(model.parameters()).device  # type: ignore[no-any-return-implicit]
    except StopIteration:
        return torch.device("cpu")


def move_to_device(batch: Any, device: torch.device) -> Any:
    r"""move_to_device(batch: Any, device: torch.device) -> Any

    Recursively move tensors in a batch to the target device.

    Handles ``dict``, ``list``, ``tuple``, ``NamedTuple``, and
    ``frozenset`` containers, recursing into nested structures.
    Non-tensor leaves are passed through unchanged.

    Review N4: the previous code did ``type(batch)(moved)`` for every
    list/tuple, which broke ``NamedTuple`` (the type's __new__ requires
    keyword args via _fields) and missed ``frozenset``.

    Args:
        batch (Any): A tensor, or a nested container of tensors (``dict``,
            ``list``, ``tuple``, ``NamedTuple``, ``frozenset``).
        device (torch.device): Target device for tensor relocation.

    Returns:
        Any: A structure of the same shape as ``batch`` with all tensors
        moved to ``device``.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    # NamedTuple check: it subclasses tuple but its __new__ takes
    # positional or keyword args. Use _make to preserve the type.
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        moved = [move_to_device(item, device) for item in batch]
        return type(batch)._make(moved)
    if isinstance(batch, frozenset):
        return frozenset(move_to_device(item, device) for item in batch)
    if isinstance(batch, (list, tuple)):
        moved = [move_to_device(item, device) for item in batch]
        return type(batch)(moved)
    return batch


__all__ = ["get_model_device", "move_to_device"]
