r"""Transformer block registry and factory functions for Qwen3.5 architectures."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Any

from torch import nn

from qwendopamine.models.blocks.reward import (
    LearnableSoftsign,
    RewardFiLM,
    RewardFourierEncoder,
    RewardStatisticsExtractor,
)


# Lazy imports to avoid circular dependency (reinforced <-> blocks)
def _lazy_grn() -> tuple[type, type]:
    from qwendopamine.models.reinforced.delta import GatedRewardNet as _GRN
    from qwendopamine.models.reinforced.delta import GatedRewardNetConfig as _GRNConfig

    return _GRN, _GRNConfig


def _lazy_value_ema() -> type:
    from qwendopamine.models.reinforced.delta import ValueBaselineEMA as _V

    return _V


def _lazy_adv_gate() -> type:
    from qwendopamine.models.reinforced.delta import AdvantageGate as _A

    return _A


def _lazy_delta_core() -> type:
    from qwendopamine.models.reinforced.delta import DeltaMemoryCore as _D

    return _D


def _lazy_reinforced() -> type:
    from qwendopamine.models.reinforced.delta import ReinforcedDeltaLayer as _R

    return _R


class _LazyBlockRegistry:
    """Dict-like registry that populates model-specific blocks on first access.

    This avoids importing concrete model implementations at module load time,
    breaking the circular dependency between ``blocks`` and ``gdn2``/``infinidopamine``/``qwen35``.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, type] = {}
        self._populated = False

    def _populate(self) -> None:
        if self._populated:
            return
        from qwendopamine.models.gdn2 import GatedDeltaNet2
        from qwendopamine.models.infinidopamine import (
            InfiniDopamineDecoderLayer,
            InfiniDopamineGatedDeltaNet,
            InfiniDopamineGatedRewardNet,
        )
        from qwendopamine.models.qwen35.modular_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5GatedDeltaNet,
        )

        self._blocks.update(
            {
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
        )
        self._populated = True

    def __getitem__(self, key: str) -> type:
        self._populate()
        return self._blocks[key]

    def __contains__(self, key: object) -> bool:
        # NOTE: This triggers _populate(), which imports gdn2, infinidopamine,
        # and qwen35. Prefer checking against an explicitly populated registry
        # if you want to avoid the import-time side effect.
        self._populate()
        return key in self._blocks

    def __iter__(self) -> Iterator[str]:
        self._populate()
        return iter(self._blocks)

    def keys(self) -> KeysView[str]:
        self._populate()
        return self._blocks.keys()

    def values(self) -> ValuesView[type]:
        self._populate()
        return self._blocks.values()

    def items(self) -> ItemsView[str, type]:
        self._populate()
        return self._blocks.items()

    def __len__(self) -> int:
        self._populate()
        return len(self._blocks)

    def __repr__(self) -> str:
        self._populate()
        return repr(self._blocks)


BLOCKS = _LazyBlockRegistry()


def build_block(block_type: str, config: Any, layer_idx: int) -> nn.Module:
    r"""Instantiate a registered block module by registry name."""
    if block_type in ("gated_reward_net", "grn"):
        grn_cls, grn_config_cls = _lazy_grn()
        hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 2048))
        return grn_cls(
            grn_config_cls(
                hidden_size=hidden_size,
                layer_idx=layer_idx,
            )
        )
    lazy_map = {
        "value_baseline_ema": _lazy_value_ema,
        "advantage_gate": _lazy_adv_gate,
        "delta_memory_core": _lazy_delta_core,
        "reinforced_delta": _lazy_reinforced,
    }
    non_layer_blocks = {
        "reward_stats_extractor",
        "reward_fourier_encoder",
        "reward_film",
        "learnable_softsign",
    }
    if block_type in lazy_map or block_type in non_layer_blocks:
        raise KeyError(
            f"Block '{block_type}' is a component module requiring explicit constructor args, "
            f"not generic build_block(config, layer_idx)."
        )
    if block_type not in BLOCKS:
        # also check lazy keys for error message
        all_keys = list(BLOCKS.keys()) + list(lazy_map.keys())
        raise KeyError(f"Unknown block type: {block_type}. Available: {all_keys}")
    return BLOCKS[block_type](config, layer_idx)
