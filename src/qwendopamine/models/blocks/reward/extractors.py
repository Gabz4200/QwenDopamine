"""Reward-specific extractor and FiLM modules — re-export shim.

The classes are split across per-concern modules:

  - :mod:`.statistics`        — :class:`RewardStatisticsExtractor`
  - :mod:`.fourier_encoder`   — :class:`RewardFourierEncoder`
  - :mod:`.reward_film`       — :class:`RewardFiLM`
"""

from qwendopamine.models.blocks.reward.fourier_encoder import RewardFourierEncoder
from qwendopamine.models.blocks.reward.reward_film import RewardFiLM
from qwendopamine.models.blocks.reward.statistics import RewardStatisticsExtractor

__all__ = [
    "RewardFiLM",
    "RewardFourierEncoder",
    "RewardStatisticsExtractor",
]
