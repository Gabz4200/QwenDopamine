# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class Tokenizer:
    """Tokenizer wrapper for GDN-2 compatibility."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        from qwendopamine.integrations.tokenizer import load_qwen35_tokenizer

        self.hf_tokenizer = load_qwen35_tokenizer(str(checkpoint_dir))

    @property
    def vocab_size(self) -> int:
        return getattr(self.hf_tokenizer, "vocab_size", 32000)

    def encode(self, string: str) -> list[int]:
        return self.hf_tokenizer.encode(string)  # type: ignore[attr-defined, no-any-return]

    def decode(self, ids: list[int]) -> str:
        return self.hf_tokenizer.decode(ids)  # type: ignore[attr-defined, no-any-return]


__all__ = ["Tokenizer"]
