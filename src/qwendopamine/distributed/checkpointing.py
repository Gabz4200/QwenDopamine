from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, **kwargs: Any) -> None:
    r"""Save model and optimizer state dicts to disk.

    Parent directories are created automatically.

    Args:
        model (torch.nn.Module): model to save.
        optimizer (torch.optim.Optimizer): optimizer to save.
        path (Path): output checkpoint path.
        **kwargs: extra entries merged into the checkpoint dict.
    """


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, map_location: str = "cpu") -> dict[str, Any]:
    r"""Load a checkpoint and restore model/optimizer state.

    Args:
        model (torch.nn.Module): model to restore.
        optimizer (torch.optim.Optimizer): optimizer to restore.
        path (Path): checkpoint path.
        map_location (str): device mapping for loading. Default: ``"cpu"``.

    Returns:
        dict[str, Any]: full checkpoint dict.
    """
