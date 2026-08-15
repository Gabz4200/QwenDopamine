from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_safetensors(
    state_dict: dict[str, torch.Tensor], path: Path, **kwargs: Any
) -> None:
    r"""Save a state dict to a safetensors file.

    Parent directories are created automatically.

    Args:
        state_dict (dict[str, Tensor]): mapping from tensor names to tensors.
        path (Path): output file path.
        **kwargs: extra keyword arguments forwarded to
            :func:`safetensors.torch.save_file`.
    """
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(path), **kwargs)


def load_safetensors(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    r"""Load a safetensors file into a state dict.

    Args:
        path (Path): path to a safetensors file.
        device (str): device to map tensors onto. Default: ``"cpu"``.

    Returns:
        dict[str, Tensor]: loaded state dict.
    """
    from safetensors.torch import load_file

    return load_file(str(path), device=device)
