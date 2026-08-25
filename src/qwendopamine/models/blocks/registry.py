r"""Transformer block registry and factory functions for Qwen3.5 architectures."""

from __future__ import annotations

from typing import Any

from torch import nn

from qwendopamine.models.blocks.reward import (
    LearnableSoftsign,
    RewardFiLM,
    RewardFourierEncoder,
    RewardStatisticsExtractor,
)
from qwendopamine.models.gdn2.gdn2 import GatedDeltaNet2
from qwendopamine.models.infinidopamine import (
    InfiniDopamineDecoderLayer,
    InfiniDopamineGatedDeltaNet,
    InfiniDopamineGatedRewardNet,
)
from qwendopamine.models.qwen35.modular_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
)


# Lazy imports to avoid circular dependency (reinforced_delta <-> blocks)
def _lazy_grn() -> type:
    from qwendopamine.models.gdn2.reinforced_delta import GatedRewardNet as _GRN
    return _GRN

def _lazy_value_ema() -> type:
    from qwendopamine.models.gdn2.reinforced_delta import ValueBaselineEMA as _V
    return _V

def _lazy_adv_gate() -> type:
    from qwendopamine.models.gdn2.reinforced_delta import AdvantageGate as _A
    return _A

def _lazy_delta_core() -> type:
    from qwendopamine.models.gdn2.reinforced_delta import DeltaMemoryCore as _D
    return _D

def _lazy_reinforced() -> type:
    from qwendopamine.models.gdn2.reinforced_delta import ReinforcedDeltaLayer as _R
    return _R

BLOCKS: dict[str, type] = {
    "gdn": Qwen3_5GatedDeltaNet,
    "gdn2": GatedDeltaNet2,
    "infini": InfiniDopamineDecoderLayer,
    "infini_gdn": InfiniDopamineGatedDeltaNet,
    "infini_reward": InfiniDopamineGatedRewardNet,
    "infinidopamine": InfiniDopamineDecoderLayer,
    "infinidopamine_decoder": InfiniDopamineDecoderLayer,
    "infinidopamine_gdn": InfiniDopamineGatedDeltaNet,
    "infinidopamine_reward": InfiniDopamineGatedRewardNet,
    "infinidopamine_grn": InfiniDopamineGatedRewardNet,
    "qwen": Qwen3_5DecoderLayer,
    "qwen35": Qwen3_5DecoderLayer,
    "qwen35_gdn": Qwen3_5GatedDeltaNet,
    "qwen35_gdn2": GatedDeltaNet2,
    "reward_stats_extractor": RewardStatisticsExtractor,
    "reward_fourier_encoder": RewardFourierEncoder,
    "reward_film": RewardFiLM,
    "learnable_softsign": LearnableSoftsign,
}


def build_block(block_type: str, config: Any, layer_idx: int) -> nn.Module:
    r"""build_block(block_type, config, layer_idx) -> nn.Module

    Instantiates a registered block module by registry name.

    Args:
        block_type (str): Key string identifying registered block class in ``BLOCKS``.
        config (Any): Configuration object containing architecture hyperparameters.
        layer_idx (int): Zero-indexed layer position integer.

    Returns:
        nn.Module: Instantiated transformer layer or module.

    Raises:
        KeyError: If ``block_type`` is not present in ``BLOCKS`` registry.
    """
    if block_type in ("gated_reward_net", "grn"):
        grn_cls = _lazy_grn()
        hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 2048))
        return grn_cls(hidden_size=hidden_size, layer_idx=layer_idx)
    lazy_map = {
        "value_baseline_ema": _lazy_value_ema,
        "advantage_gate": _lazy_adv_gate,
        "delta_memory_core": _lazy_delta_core,
        "reinforced_delta": _lazy_reinforced,
    }
    if block_type in lazy_map:
        raise KeyError(f"Block '{block_type}' requires explicit constructor args, not generic build_block(config, layer_idx).")
    if block_type not in BLOCKS:
        # also check lazy keys for error message
        all_keys = list(BLOCKS.keys()) + list(lazy_map.keys())
        raise KeyError(f"Unknown block type: {block_type}. Available: {all_keys}")
    return BLOCKS[block_type](config, layer_idx)
