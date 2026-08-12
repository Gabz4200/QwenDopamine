from __future__ import annotations

from typing import Any

from .qwen_block import QwenDecoderLayer
from .gdn_block import GatedDeltaNetBlock
from .experimental_block import ExperimentalBlock

BLOCKS: dict[str, type] = {
    "qwen": QwenDecoderLayer,
    "gdn": GatedDeltaNetBlock,
    "experimental": ExperimentalBlock,
}


def build_block(block_type: str, config: Any, layer_idx: int) -> Any:
    if block_type not in BLOCKS:
        raise KeyError(f"Unknown block type: {block_type}. Available: {list(BLOCKS)}")
    return BLOCKS[block_type](config, layer_idx)
