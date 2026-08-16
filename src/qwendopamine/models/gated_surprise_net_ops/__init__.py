r"""
Triton GPU ops for GatedSurpriseNet.

Provides custom Triton chunkwise parallel kernels for precision-weighted
GatedSurpriseNet fast-weight memory recurrence, incorporating the per-value-channel
precision metric \pi_t \in \mathbb{R}^{d_v} into the 3D (d_v x C x C) WY lower-triangular solve.
"""

from __future__ import annotations

from .chunk_gated_surprise_net import chunk_gated_surprise_net

__all__ = ["chunk_gated_surprise_net"]
