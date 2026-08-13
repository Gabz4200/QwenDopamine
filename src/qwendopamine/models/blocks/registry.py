"""Transformer block registry."""

from __future__ import annotations

from typing import Any

from .experimental_block import ExperimentalBlock
from .gdn2 import GatedDeltaNet2Block
from .gdn_block import GatedDeltaNetBlock
from .qwen_block import QwenDecoderLayer

BLOCKS: dict[str, type] = {
    "qwen": QwenDecoderLayer,
    "gdn": GatedDeltaNetBlock,
    "gdn2": GatedDeltaNet2Block,
    "experimental": ExperimentalBlock,
}


def build_block(block_type: str, config: Any, layer_idx: int) -> Any:
    r"""Instantiate a registered block by name.

    Args:
        block_type (str): registered block key, such as ``"qwen"`` or ``"gdn"``.
        config: any object with block-level attributes forwarded to the block constructor.
        layer_idx (int): layer index passed to the block constructor.

    Returns:
        Any: instantiated :class:`torch.nn.Module` block.

    Raises:
        KeyError: if ``block_type`` is not registered in :data:`BLOCKS`.
    """
    if block_type not in BLOCKS:
        raise KeyError(f"Unknown block type: {block_type}. Available: {list(BLOCKS)}")
    return BLOCKS[block_type](config, layer_idx)
