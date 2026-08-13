"""Transformer block registry."""

from qwendopamine.models.blocks.gdn2 import (  # noqa: F401
    GDN2Mixer,
    GDN2Projections,
    GatedDeltaNet2Block,
)
from qwendopamine.models.blocks.gdn2_ops import dispatch_gdn2  # noqa: F401
from qwendopamine.models.blocks.registry import BLOCKS, build_block  # noqa: F401

__all__ = [
    "BLOCKS",
    "build_block",
    "GDN2Mixer",
    "GDN2Projections",
    "GatedDeltaNet2Block",
    "dispatch_gdn2",
]
