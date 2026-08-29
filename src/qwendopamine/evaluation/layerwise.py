r"""Layerwise activation statistics evaluation utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.utils import get_model_device, move_to_device


def layerwise_stats(
    model: nn.Module, dataloader: Any, max_steps: int = 50
) -> dict[str, float]:
    r"""Collect per-layer activation statistics across forward passes."""
    model.eval()
    device = get_model_device(model)
    layer_sums: dict[str, float] = {}
    layer_counts: dict[str, int] = {}
    hooks = []

    def _make_hook(name: str) -> Any:
        def hook(mod: nn.Module, inp: Any, out: Any) -> None:
            tensor = out[0] if isinstance(out, tuple) else out
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                norm = float(tensor.detach().norm().item())
                mean = float(tensor.detach().mean().item())
                layer_sums[f"{name}.norm"] = layer_sums.get(f"{name}.norm", 0.0) + norm
                layer_counts[f"{name}.norm"] = layer_counts.get(f"{name}.norm", 0) + 1
                layer_sums[f"{name}.mean"] = layer_sums.get(f"{name}.mean", 0.0) + mean
                layer_counts[f"{name}.mean"] = layer_counts.get(f"{name}.mean", 0) + 1

        return hook

    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if step >= max_steps:
                    break
                batch = move_to_device(batch, device)
                model(**batch)
    finally:
        for h in hooks:
            h.remove()

    return {k: layer_sums[k] / max(layer_counts[k], 1) for k in layer_sums}


__all__ = ["layerwise_stats"]
