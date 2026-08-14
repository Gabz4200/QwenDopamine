"""Transformer block registry for Qwen3.5."""

from __future__ import annotations

from typing import Any

from torch import nn

from qwendopamine.models.qwen35.modular_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
)

BLOCKS: dict[str, type] = {
    "qwen": Qwen3_5DecoderLayer,
    "gdn": Qwen3_5GatedDeltaNet,
    "qwen35": Qwen3_5DecoderLayer,
    "qwen35_gdn": Qwen3_5GatedDeltaNet,
}


def build_block(block_type: str, config: Any, layer_idx: int) -> nn.Module:
    r"""Instantiate a registered block by name."""
    if block_type not in BLOCKS:
        raise KeyError(f"Unknown block type: {block_type}. Available: {list(BLOCKS)}")
    return BLOCKS[block_type](config, layer_idx)
