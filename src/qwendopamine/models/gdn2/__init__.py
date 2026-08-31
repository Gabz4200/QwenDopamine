# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

# Re-export new submodule symbols for backward compatibility
from .backend import GDN2_BACKENDS, resolve_gdn2_backend
from .chunk import (
    compute_gdn2_intra_chunk_scores,
    compute_gdn2_wy_coefficients,
)
from .config import GDN2Config
from .gdn2 import (
    GatedDeltaNet2,
    RMSNormGated,
    RMSNormGatedNoCast,
    ShortConvolution,
    torch_chunk_gdn2,
    torch_recurrent_gdn2,
)
from .host import Block, GDN2Host
from .reinforced_delta import (
    AdvantageGate,
    DeltaMemoryCore,
    GatedRewardNet,
    ReinforcedDeltaLayer,
    ValueBaselineEMA,
)

__all__ = [
    "GDN2_BACKENDS",
    "AdvantageGate",
    "Block",
    "DeltaMemoryCore",
    "GDN2Config",
    "GDN2Host",
    "GatedDeltaNet2",
    "GatedRewardNet",
    "RMSNormGated",
    "RMSNormGatedNoCast",
    "ReinforcedDeltaLayer",
    "ShortConvolution",
    "ValueBaselineEMA",
    "compute_gdn2_intra_chunk_scores",
    "compute_gdn2_wy_coefficients",
    "resolve_gdn2_backend",
    "torch_chunk_gdn2",
    "torch_recurrent_gdn2",
]
