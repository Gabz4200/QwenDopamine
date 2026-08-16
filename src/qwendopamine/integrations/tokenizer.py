r"""Tokenizer loading utilities for Qwen architectures."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


def load_qwen35_tokenizer(model_name: str, **kwargs: Any) -> AutoTokenizer:
    r"""load_qwen35_tokenizer(model_name, **kwargs) -> AutoTokenizer

    Loads a Qwen3.5 tokenizer with automatic GGUF repo resolution and fallback candidates.

    Args:
        model_name (str): Hugging Face repository ID or local GGUF file path.
        **kwargs (Any): Additional keyword arguments passed to :meth:`transformers.AutoTokenizer.from_pretrained`.

    Returns:
        AutoTokenizer: Loaded tokenizer instance.

    Raises:
        RuntimeError: If all candidate repositories fail to load a valid tokenizer.

    Examples::

        >>> tokenizer = load_qwen35_tokenizer("Qwen/Qwen3.5-0.8B")
    """
    if model_name.endswith(".gguf"):
        model_name = model_name.rsplit("/", 1)[0]

    repo_candidates = [model_name, "Qwen/Qwen3.5-0.8B"]

    last_error: Exception | None = None
    for candidate in repo_candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, **kwargs)  # type: ignore[return-value]
        except (OSError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(
        "Failed to load Qwen3.5 tokenizer from candidate repos. "
        f"Last error: {last_error}"
    )


__all__ = ["load_qwen35_tokenizer"]
