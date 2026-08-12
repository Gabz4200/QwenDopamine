from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), **kwargs}, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, map_location: str = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
