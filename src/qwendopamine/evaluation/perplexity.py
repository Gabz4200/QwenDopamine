r"""Perplexity computation utilities for causal language models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.utils import get_model_device


def compute_perplexity(
    model: nn.Module, dataloader: Any, max_steps: int = 500
) -> float:
    r"""compute_perplexity(model, dataloader, max_steps=500) -> float

    Estimates perplexity over a dataloader sequence by accumulating token-weighted cross-entropy loss.

    .. math::
        \text{PPL} = \exp\left( \frac{1}{N} \sum_{i=1}^N \mathcal{L}_i \right)

    Args:
        model (nn.Module): Causal language model returning scalar loss or loss dict.
        dataloader (Any): Iterable dataloader yielding batch dictionaries with ``input_ids`` tensors.
        max_steps (int, optional): Maximum number of evaluation steps to compute over. Default: ``500``.

    Returns:
        float: Exponentiated average per-token cross-entropy loss value.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = get_model_device(model)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
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

    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()


__all__ = ["compute_perplexity"]
