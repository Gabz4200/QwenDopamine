# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization in pure PyTorch."""

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(self.dim, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * normed

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


# FusedRMSNorm alias for hardware-agnostic execution
FusedRMSNorm = RMSNorm

__all__ = ["FusedRMSNorm", "RMSNorm"]
