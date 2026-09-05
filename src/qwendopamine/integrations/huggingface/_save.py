"""Save helpers for QwenDopamine HF models.

Extracted from :mod:`integration` for modularity.
"""

from __future__ import annotations

from typing import Any


def save_model(model: Any, save_directory: str) -> None:
    r"""save_model(model, save_directory) -> None

    Save a model with ``save_pretrained``.

    Args:
        model (Any): Hugging Face model instance.
        save_directory (str): output directory path.
    """
    model.save_pretrained(save_directory)


__all__ = ["save_model"]
