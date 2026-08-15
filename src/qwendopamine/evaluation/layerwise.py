from __future__ import annotations

from typing import Any

import torch

from qwendopamine.utils import get_model_device as _get_model_device


def layerwise_stats(
    model: torch.nn.Module, dataloader: Any, max_steps: int = 50
) -> dict[str, float]:
    r"""Collect per-layer activation statistics from one or more forward passes.

    Runs up to ``max_steps`` batches through the model under ``torch.no_grad``
    and returns a placeholder stats dict. Extend this hook to capture
    intermediate activations when adding layerwise instrumentation.

    Args:
        model (torch.nn.Module): model to inspect.
        dataloader (Any): iterable yielding input batches.
        max_steps (int): maximum number of batches to process. Default: ``50``.

    Returns:
        dict[str, float]: collected layer statistics. Currently empty.
    """
    model.eval()
    stats: dict[str, float] = {}
    device = _get_model_device(model)
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
