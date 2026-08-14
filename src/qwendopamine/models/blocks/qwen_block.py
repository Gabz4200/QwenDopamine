"""Qwen3.5 decoder layer block."""

from __future__ import annotations

from qwendopamine.models.qwen35.modular_qwen3_5 import Qwen3_5DecoderLayer

# Backwards compatibility alias
QwenDecoderLayer = Qwen3_5DecoderLayer

__all__ = [
    "Qwen3_5DecoderLayer",
    "QwenDecoderLayer",
]
