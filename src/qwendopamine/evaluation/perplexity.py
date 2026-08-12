from __future__ import annotations

from typing import Any

import torch


def compute_perplexity(model: torch.nn.Module, dataloader: Any, max_steps: int = 500) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break
            batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
            total_loss += loss.item() * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()

    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
