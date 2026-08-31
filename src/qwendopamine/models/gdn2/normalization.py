# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Normalization layers for GDN-2.

This module defines the RMSNorm-based gated normalization used in the GDN-2
output branch. The no-cast variant preserves the input dtype throughout the
forward pass, which matters for bf16 training stability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNormGatedNoCast(nn.Module):
    """SiLU-gated RMS Normalization without dtype promotion.

    Computes the normalization entirely in the input dtype. This matches the
    reference GDN-2 implementation and avoids the ~3e-2 bf16 drift that occurs
    when promoting to float32 intermediate results.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps) * self.weight
        return normed * F.silu(z)


class RMSNormGated(RMSNormGatedNoCast):
    """Deprecated alias for :class:`RMSNormGatedNoCast`.

    .. deprecated::
        Use ``RMSNormGatedNoCast`` to make the no-cast behavior explicit and
        avoid confusion with other RMSNorm gated variants in the codebase.
    """


__all__ = ["RMSNormGated", "RMSNormGatedNoCast"]
