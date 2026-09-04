r"""Tokenizer loading utilities for Qwen architectures."""

from __future__ import annotations

import os
from typing import Any

from transformers import AutoTokenizer

from qwendopamine import DEFAULT_QWEN35_REPO

_DEFAULT_QWEN35_REPO = DEFAULT_QWEN35_REPO


def load_qwen35_tokenizer(model_name: str, **kwargs: Any) -> Any:
    r"""Load a Qwen3.5 tokenizer with automatic GGUF repo resolution and fallback candidates."""
    if model_name.endswith(".gguf"):
        dirname = os.path.dirname(model_name)
        model_name = dirname if dirname else _DEFAULT_QWEN35_REPO

    repo_candidates = [model_name, _DEFAULT_QWEN35_REPO]

    last_error: Exception | None = None
    for candidate in repo_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, **kwargs)
            if tokenizer is not None:
                if (
                    getattr(tokenizer, "pad_token", None) is None
                    and getattr(tokenizer, "eos_token", None) is not None
                ):
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer
        except (OSError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(
        "Failed to load Qwen3.5 tokenizer from candidate repos. "
        f"Last error: {last_error}"
    )


__all__ = ["load_qwen35_tokenizer"]
