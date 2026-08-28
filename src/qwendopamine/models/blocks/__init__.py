"""Transformer blocks package.

The ``blocks`` package provides reward-conditioning building blocks and a
lazy block registry. Model-specific implementations are imported by the
registry on first use, so importing ``qwendopamine.models.blocks`` does not
pull in the full model tree.
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

__all__ = [
    "BLOCKS",
    "AsinhScaler",
    "LearnableFourierFeatures",
    "LearnableSoftsign",
    "RewardFiLM",
    "RewardFourierEncoder",
    "RewardStatisticsExtractor",
    "TokenWiseFiLM",
    "build_block",
]
