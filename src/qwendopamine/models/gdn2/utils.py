# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

from __future__ import annotations

from torch import nn


def find_multiple(n: int, k: int) -> int:
    """Find smallest multiple of k >= n."""
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def num_parameters(module: nn.Module, requires_grad: bool | None = None) -> int:
    """Count parameters in module."""
    return sum(
        p.numel()
        for p in module.parameters()
        if requires_grad is None or p.requires_grad == requires_grad
    )


__all__ = ["find_multiple", "num_parameters"]
