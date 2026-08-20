r"""Top-level re-exports for Qwen3.5 architecture blocks and reward blocks."""

from __future__ import annotations

from qwendopamine.models.blocks import (
    BLOCKS,
    AsinhScaler,
    GatedDeltaNet2Block,
    GatedDeltaNetBlock,
    LearnableFourierFeatures,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    QwenDecoderLayer,
    RewardEncoder,
    TokenWiseFiLM,
    build_block,
)

__all__ = [
    "BLOCKS",
    "AsinhScaler",
    "GatedDeltaNet2Block",
    "GatedDeltaNetBlock",
    "LearnableFourierFeatures",
    "Qwen3_5DecoderLayer",
    "Qwen3_5GatedDeltaNet",
    "QwenDecoderLayer",
    "RewardEncoder",
    "TokenWiseFiLM",
    "build_block",
]
