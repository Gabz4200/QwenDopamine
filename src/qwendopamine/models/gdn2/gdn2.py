# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Backward-compatible re-exports for the decomposed GDN-2 module.

The GDN-2 implementation has been split into focused submodules:

- :mod:`qwendopamine.models.gdn2.block` -- :class:`GatedDeltaNet2` main block
- :mod:`qwendopamine.models.gdn2.core` -- pure PyTorch recurrence engine
- :mod:`qwendopamine.models.gdn2.chunk` -- chunkwise WY kernel
- :mod:`qwendopamine.models.gdn2.convolution` -- :class:`ShortConvolution`
- :mod:`qwendopamine.models.gdn2.normalization` -- gated RMSNorm
- :mod:`qwendopamine.models.gdn2.cache_utils` -- padding-mask helpers
- :mod:`qwendopamine.models.gdn2.backend` -- backend resolution

All historical imports from ``qwendopamine.models.gdn2.gdn2`` continue to work
through this module. New code should import from the specific submodules or
from the ``qwendopamine.models.gdn2`` package directly.
"""

from __future__ import annotations

# Re-export the public API from the decomposed submodules.
from qwendopamine.models.gdn2.backend import (
    GDN2_BACKENDS,
    resolve_gdn2_backend,
)
from qwendopamine.models.gdn2.block import GatedDeltaNet2
from qwendopamine.models.gdn2.chunk import (
    compute_gdn2_intra_chunk_scores,
    compute_gdn2_wy_coefficients,
    torch_chunk_gdn2,
)
from qwendopamine.models.gdn2.convolution import ShortConvolution
from qwendopamine.models.gdn2.core import torch_recurrent_gdn2
from qwendopamine.models.gdn2.normalization import RMSNormGated, RMSNormGatedNoCast

__all__ = [
    "GDN2_BACKENDS",
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
