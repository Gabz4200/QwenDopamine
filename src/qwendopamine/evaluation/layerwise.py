from __future__ import annotations

from typing import Any

import torch


def layerwise_stats(model: torch.nn.Module, dataloader: Any, max_steps: int = 50) -> dict[str, float]:
    r"""Collect per-layer activation statistics from one forward pass.

    Currently a placeholder: it runs a single batch through the model and
    returns an empty stats dict. Extend this hook to inspect intermediate
    activations when adding layerwise instrumentation.

    Args:
        model (torch.nn.Module): model to inspect.
        dataloader (Any): iterable yielding input batches.
        max_steps (int): unused; kept for interface consistency. Default: ``50``.

    Returns:
        dict[str, float]: collected layer statistics. Currently empty.
    """
    model.eval()
    stats: dict[str, float] = {}
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break
            batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model(**batch)
            break
    return stats
