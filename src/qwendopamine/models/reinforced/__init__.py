"""RL-augmented memory conditioning modules.

Provides reward-conditioned fast-weight components that are orthogonal to any
specific attention primitive (GDN-2, SWA, etc.).
"""

from .delta import (
    AdvantageGate,
    DeltaMemoryCore,
    GatedRewardNet,
    GatedRewardNetConfig,
    ReinforcedDeltaLayer,
    ValueBaselineEMA,
    normalize_reward_for_advantage,
)

__all__ = [
    "AdvantageGate",
    "DeltaMemoryCore",
    "GatedRewardNet",
    "GatedRewardNetConfig",
    "ReinforcedDeltaLayer",
    "ValueBaselineEMA",
    "normalize_reward_for_advantage",
]
