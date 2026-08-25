"""Reward conditioning blocks.

General-purpose building blocks and reward-specific modules are split into
separate files for maintainability. This ``__init__`` re-exports the full
public API so existing ``from ...blocks.reward import X`` statements keep
working.
"""

from qwendopamine.models.blocks.reward.components import (
    AsinhScaler,
    LearnableFourierFeatures,
    LearnableSoftsign,
    TokenWiseFiLM,
)
from qwendopamine.models.blocks.reward.extractors import (
    RewardFiLM,
    RewardFourierEncoder,
    RewardStatisticsExtractor,
)

__all__ = [
    "AsinhScaler",
    "LearnableFourierFeatures",
    "LearnableSoftsign",
    "RewardFiLM",
    "RewardFourierEncoder",
    "RewardStatisticsExtractor",
    "TokenWiseFiLM",
]
