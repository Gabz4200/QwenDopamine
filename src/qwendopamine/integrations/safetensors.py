r"""Safetensors serialization and deserialization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_safetensors(
    state_dict: dict[str, torch.Tensor], path: Path | str, **kwargs: Any
) -> None:
    r"""Save a PyTorch state dictionary to a safetensors file."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(path), **kwargs)


def load_safetensors(path: Path | str, device: str = "cpu") -> dict[str, torch.Tensor]:
    r"""Load a safetensors file into a PyTorch state dictionary."""
    from safetensors.torch import load_file

    return load_file(str(path), device=device)


__all__ = [
    "load_safetensors",
    "save_safetensors",
]
