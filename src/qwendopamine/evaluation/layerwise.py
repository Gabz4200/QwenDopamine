r"""Layerwise activation statistics evaluation utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.utils import get_model_device


def layerwise_stats(
    model: nn.Module, dataloader: Any, max_steps: int = 50
) -> dict[str, float]:
    r"""layerwise_stats(model, dataloader, max_steps=50) -> dict[str, float]

    Processes evaluation batches to collect per-layer activation statistics across forward passes.

    Args:
        model (nn.Module): PyTorch module instance to inspect.
        dataloader (Any): Iterable dataloader yielding input batch dictionaries.
        max_steps (int, optional): Maximum evaluation steps to process. Default: ``50``.

    Returns:
        dict[str, float]: Dictionary mapping layer names to collected activation statistics.
    """
    model.eval()
    stats: dict[str, float] = {}
    device = get_model_device(model)
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            model(**batch)
    return stats


__all__ = ["layerwise_stats"]
