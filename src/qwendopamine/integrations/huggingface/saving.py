"""Saving helpers for QwenDopamine HF models.

Provides ``save_model`` entry point that delegates to
``transformers.PreTrainedModel.save_pretrained``.
"""

from __future__ import annotations

from typing import Any


def save_model(model: Any, save_directory: str) -> None:
    r"""Save a model with ``save_pretrained``.

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.save_model` static method body.

    Args:
        model (Any): Hugging Face model instance.
        save_directory (str): output directory path.
    """
    model.save_pretrained(save_directory)
