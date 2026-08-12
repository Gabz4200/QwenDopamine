from __future__ import annotations

from typing import Any

import torch


def layerwise_stats(model: torch.nn.Module, dataloader: Any, max_steps: int = 50) -> dict[str, float]:
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
