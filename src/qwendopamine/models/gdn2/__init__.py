# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

from .backend import GDN2_BACKENDS, resolve_gdn2_backend
from .block import GatedDeltaNet2
from .config import GDN2Config
from .host import Block, GDN2Host
from .ops import RMSNormGated, RMSNormGatedNoCast, ShortConvolution
from .recurrence import (
    compute_gdn2_intra_chunk_scores,
    compute_gdn2_wy_coefficients,
    torch_chunk_gdn2,
    torch_recurrent_gdn2,
)

__all__ = [
    "GDN2_BACKENDS",
    "Block",
    "GDN2Config",
    "GDN2Host",
    "GatedDeltaNet2",
    "RMSNormGated",
    "RMSNormGatedNoCast",
    "ShortConvolution",
    "compute_gdn2_intra_chunk_scores",
    "compute_gdn2_wy_coefficients",
    "resolve_gdn2_backend",
    "torch_chunk_gdn2",
    "torch_recurrent_gdn2",
]
