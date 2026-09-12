r"""Tokenizer loading utilities for Qwen architectures."""

from __future__ import annotations

import os
from typing import Any

from transformers import AutoTokenizer

from qwendopamine import DEFAULT_QWEN35_REPO

_DEFAULT_QWEN35_REPO = DEFAULT_QWEN35_REPO


def load_qwen35_tokenizer(model_name: str, **kwargs: Any) -> Any:
    r"""Load a Qwen3.5 tokenizer from the given model name."""
    if model_name.endswith(".gguf"):
        dirname = os.path.dirname(model_name)
        model_name = dirname if dirname else _DEFAULT_QWEN35_REPO

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if (
        tokenizer is not None
        and getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


__all__ = ["load_qwen35_tokenizer"]
