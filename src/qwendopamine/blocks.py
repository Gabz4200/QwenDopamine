r"""Top-level re-exports for Qwen3.5 architecture blocks and reward blocks."""

from __future__ import annotations

from qwendopamine.models.blocks import (
    BLOCKS,
    AdaLN,
    FourierFeatures,
    GatedDeltaNet2Block,
    GatedDeltaNetBlock,
    GatedSurpriseNetAdam,
    GatedSurpriseNetBlock,
    LearnableFourierFeatures,
    PositionalEncoding,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    QwenDecoderLayer,
    RewardEncoder,
    build_block,
)

__all__ = [
    "BLOCKS",
    "AdaLN",
    "FourierFeatures",
    "GatedDeltaNet2Block",
    "GatedDeltaNetBlock",
    "GatedSurpriseNetAdam",
    "GatedSurpriseNetBlock",
    "LearnableFourierFeatures",
    "PositionalEncoding",
    "Qwen3_5DecoderLayer",
    "Qwen3_5GatedDeltaNet",
    "QwenDecoderLayer",
    "RewardEncoder",
    "build_block",
]
