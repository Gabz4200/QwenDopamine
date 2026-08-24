"""Transformer blocks package.

Block classes and aliases are consolidated here so the package is the single
point of entry for transformer layers, reward conditioning blocks, and the
block registry. `gdn_block.py` and `qwen_block.py` were thin alias-only
modules and are merged directly into this init.
"""

from qwendopamine.models.blocks.registry import BLOCKS, build_block
from qwendopamine.models.blocks.reward import (
    AsinhScaler,
    LearnableFourierFeatures,
    LearnableSoftsign,
    RewardFiLM,
    RewardFourierEncoder,
    RewardStatisticsExtractor,
    TokenWiseFiLM,
)
from qwendopamine.models.gdn2.gdn2 import GatedDeltaNet2
from qwendopamine.models.qwen35.modular_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
)

# Aliases (previously defined in gdn_block.py / qwen_block.py).
GatedDeltaNetBlock = Qwen3_5GatedDeltaNet
GatedDeltaNet2Block = GatedDeltaNet2
QwenDecoderLayer = Qwen3_5DecoderLayer

__all__ = [
    "BLOCKS",
    "AsinhScaler",
    "GatedDeltaNet2Block",
    "GatedDeltaNetBlock",
    "LearnableFourierFeatures",
    "LearnableSoftsign",
    "Qwen3_5DecoderLayer",
    "Qwen3_5GatedDeltaNet",
    "QwenDecoderLayer",
    "RewardFiLM",
    "RewardFourierEncoder",
    "RewardStatisticsExtractor",
    "TokenWiseFiLM",
    "build_block",
]
