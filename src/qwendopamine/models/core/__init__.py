"""Shared model primitives used across all model families.

Each module here is a small, dependency-free building block (embeddings,
normalization, output projection, config adapter) that every architecture
under :mod:`qwendopamine.models.*` consumes.
"""

from qwendopamine.models.core.config_adapter import ConfigAdapter
from qwendopamine.models.core.embeddings import PositionEmbeddings, TokenEmbeddings
from qwendopamine.models.core.normalization import (
    RMSNorm,
    RMSNormGated,
    apply_mask_to_padding_states,
)
from qwendopamine.models.core.output_head import LMHead

__all__ = [
    "ConfigAdapter",
    "LMHead",
    "PositionEmbeddings",
    "RMSNorm",
    "RMSNormGated",
    "TokenEmbeddings",
    "apply_mask_to_padding_states",
]
