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
from qwendopamine.models.blocks.reward import (
    AdaLN,
    FourierFeatures,
    LearnableFourierFeatures,
    PositionalEncoding,
    RewardEncoder,
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
