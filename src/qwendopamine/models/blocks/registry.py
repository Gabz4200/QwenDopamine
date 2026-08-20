r"""Transformer block registry and factory functions for Qwen3.5 architectures."""

from __future__ import annotations

from typing import Any

from torch import nn

from qwendopamine.models.blocks.reward import RewardEncoder
from qwendopamine.models.gdn2.gdn2 import GatedDeltaNet2
from qwendopamine.models.qwen35.modular_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
)

BLOCKS: dict[str, type] = {
    "gdn": Qwen3_5GatedDeltaNet,
    "gdn2": GatedDeltaNet2,
    "qwen": Qwen3_5DecoderLayer,
    "qwen35": Qwen3_5DecoderLayer,
    "qwen35_gdn": Qwen3_5GatedDeltaNet,
    "qwen35_gdn2": GatedDeltaNet2,
    "reward_encoder": RewardEncoder,
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
    if block_type not in BLOCKS:
        raise KeyError(f"Unknown block type: {block_type}. Available: {list(BLOCKS)}")
    return BLOCKS[block_type](config, layer_idx)
