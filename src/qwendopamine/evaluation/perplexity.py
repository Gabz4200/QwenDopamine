r"""Perplexity computation utilities for causal language models."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn

from qwendopamine.utils import get_model_device, move_to_device


def compute_perplexity(
    model: nn.Module, dataloader: Any, max_steps: int = 500
) -> float:
    r"""Estimate perplexity over a dataloader by accumulating token-weighted cross-entropy loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = get_model_device(model)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break
            batch = move_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            if "labels" in batch:
                num_tokens = int((batch["labels"] != -100).sum().item())
            elif "attention_mask" in batch:
                num_tokens = int(batch["attention_mask"].sum().item())
            else:
                num_tokens = batch["input_ids"].numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    if ppl == float("inf"):
        warnings.warn(
            f"Perplexity overflowed (avg_loss={avg_loss:.2f}); returning float('inf')."
        )
    return ppl


__all__ = ["compute_perplexity"]
