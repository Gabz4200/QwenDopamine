r"""Safetensors serialization and deserialization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_safetensors(
    state_dict: dict[str, torch.Tensor], path: Path, **kwargs: Any
) -> None:
    r"""save_safetensors(state_dict, path, **kwargs) -> None

    Saves a PyTorch state dictionary to a safetensors file.

    Parent directories are created automatically if missing.

    Args:
        state_dict (dict[str, Tensor]): State dictionary mapping parameter names to tensors.
        path (Path): Target output file path.
        **kwargs (Any): Additional keyword arguments passed to :func:`safetensors.torch.save_file`.

    Examples::

        >>> from pathlib import Path
        >>> state_dict = {"weight": torch.randn(2, 2)}
        >>> save_safetensors(state_dict, Path("/tmp/model.safetensors"))
    """
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(path), **kwargs)


def load_safetensors(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    r"""load_safetensors(path, device="cpu") -> dict[str, Tensor]

    Loads a safetensors file into a PyTorch state dictionary mapped to the target device.

    Args:
        path (Path): Source safetensors file path.
        device (str, optional): Target device mapping string. Default: ``"cpu"``.

    Returns:
        dict[str, Tensor]: Loaded state dictionary mapping tensor names to tensors.

    Examples::

        >>> from pathlib import Path
        >>> state_dict = load_safetensors(Path("/tmp/model.safetensors"), device="cpu")
    """
    from safetensors.torch import load_file

    return load_file(str(path), device=device)


__all__ = [
    "load_safetensors",
    "save_safetensors",
]
