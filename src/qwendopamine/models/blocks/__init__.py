"""Transformer blocks package."""

from qwendopamine.models.blocks.gdn_block import (
    GatedDeltaNet2Block,
    GatedDeltaNetBlock,
    GatedSurpriseNetAdam,
    GatedSurpriseNetBlock,
    Qwen3_5GatedDeltaNet,
)
from qwendopamine.models.blocks.qwen_block import (
    Qwen3_5DecoderLayer,
    QwenDecoderLayer,
)
from qwendopamine.models.blocks.registry import BLOCKS, build_block

__all__ = [
    "BLOCKS",
    "GatedDeltaNet2Block",
    "GatedDeltaNetBlock",
    "GatedSurpriseNetAdam",
    "GatedSurpriseNetBlock",
    "Qwen3_5DecoderLayer",
    "Qwen3_5GatedDeltaNet",
    "QwenDecoderLayer",
    "build_block",
]
