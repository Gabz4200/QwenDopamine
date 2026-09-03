"""Taichi-accelerated kernels for Gated DeltaNet-2 (GDN-2).

This module replaces the previous Triton / flash-linear-attention CUDA path
with custom Taichi kernels that JIT-compile to native CPU code or GPU
shaders depending on the runtime hardware. The same Taichi source produces
both targets, so no CPU-only fallback is needed: Taichi handles dispatch.

Public API:
    - :func:`chunk_taichi_gdn2`        -- parallel chunkwise training path.
    - :func:`recurrent_taichi_gdn2`    -- token-by-token recurrent path.
    - :func:`is_available`             -- True when Taichi is importable.
    - :func:`taichi_arch`              -- active backend (``cpu`` / ``cuda`` / ``gpu``).

All entrypoints are drop-in replacements for the previous pure-PyTorch
``torch_chunk_gdn2`` / ``torch_recurrent_gdn2`` functions and the legacy
``triton.chunk_gdn2`` / ``triton.fused_recurrent_gdn2`` modules.
"""

from __future__ import annotations

from qwendopamine.models.gdn2.taichi.api import (
    chunk_taichi_gdn2,
    recurrent_taichi_gdn2,
)
from qwendopamine.models.gdn2.taichi.runtime import (
    is_available,
    taichi_arch,
)

__all__ = [
    "chunk_taichi_gdn2",
    "is_available",
    "recurrent_taichi_gdn2",
    "taichi_arch",
]