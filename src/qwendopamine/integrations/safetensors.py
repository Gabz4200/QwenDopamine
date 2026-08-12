from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_safetensors(state_dict: dict[str, torch.Tensor], path: Path, **kwargs: Any) -> None:
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(path), **kwargs)


def load_safetensors(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(str(path), device=device)
