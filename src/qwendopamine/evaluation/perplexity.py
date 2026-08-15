from __future__ import annotations

from typing import Any

import torch

from qwendopamine.utils import get_model_device


def compute_perplexity(
    model: torch.nn.Module, dataloader: Any, max_steps: int = 500
) -> float:
    r"""Estimate perplexity over a dataloader.

    Accumulates cross-entropy loss token-wise and exponentiates the average
    negative log-likelihood. The model is put in eval mode and no gradients
    are computed.

    Args:
        model (torch.nn.Module): causal language model that returns ``loss``
            under ``**batch`` or as ``outputs.loss``.
        dataloader (Any): iterable yielding dicts with ``input_ids`` tensors.
        max_steps (int): maximum number of batches to evaluate. Default: ``500``.

    Returns:
        float: exponentiated average loss as perplexity.
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
            total_loss += loss.item() * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()

    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
