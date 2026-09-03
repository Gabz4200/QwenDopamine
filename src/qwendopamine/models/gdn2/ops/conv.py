# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Depthwise 1D causal convolution with decoding-state cache.

This module provides :class:`ShortConvolution`, the local pre-filter used in
GDN-2 to inject n-gram inductive bias into the Q, K, and V projections before
the matrix-valued recurrent state update. It supports both parallel sequence
processing (training) and single-step auto-regressive decoding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ShortConvolution(nn.Module):
    r"""Pure PyTorch depthwise 1D short convolution with causal padding and cache.

    This module provides the local pre-filter used in GDN-2 to inject n-gram
    inductive bias into the Q, K, and V projections before the matrix-valued
    recurrent state update. It supports both parallel sequence processing
    (training) and single-step auto-regressive decoding.

    The causal ``Conv1d`` has ``groups=hidden_size`` (depthwise) and
    ``padding=kernel_size-1`` so that the output at time ``t`` depends only on
    inputs up to time ``t`` (inclusive).

    Args:
        hidden_size: Dimension of the hidden states ``D`` (also the Conv1d
            ``in_channels``/``out_channels``).
        kernel_size: Size of the causal convolution kernel. Default ``4``.
        bias: If ``True``, add a learnable bias to the convolution.
        activation: Activation function applied after the convolution.
            Default ``"silu"``.

    Returns:
        A tuple ``(out, new_cache)`` where ``out`` is the filtered output
        tensor ``[B, T, D]`` and ``new_cache`` is the optional KV cache
        tensor of shape ``[B, D, kernel_size-1]`` for the next step.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        bias: bool = False,
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Depthwise causal 1D convolution forward pass.

        Args:
            x: Input tensor ``[B, T, D]``.
            cache: Optional KV cache ``[B, D, kernel_size-1]`` for single-step
                autoregressive decoding. When provided and ``T == 1``, the cache
                is concatenated with the new token before convolution.
            output_final_state: If ``True``, return the last ``kernel_size-1``
                input channels as ``new_cache`` for the next step.
            cu_seqlens: Unused (reserved for packed-sequence interface compatibility).

        Returns:
            ``(out, new_cache)`` where ``out`` is ``[B, T, D]`` and ``new_cache``
            is ``[B, D, kernel_size-1]`` when ``output_final_state`` is ``True``.
        """
        _, t, d = x.shape
        x_t = x.transpose(1, 2)  # [B, D, T]

        if cache is None and output_final_state and t == 1:
            cache = torch.zeros(
                x.shape[0], d, self.kernel_size - 1, device=x.device, dtype=x.dtype
            )

        if cache is not None and t == 1:
            x_cat = torch.cat([cache, x_t], dim=-1)
            new_cache = x_cat[:, :, 1:] if output_final_state else None
            out = F.conv1d(x_cat, self.conv1d.weight, self.conv1d.bias, groups=d)
        else:
            new_cache = (
                x_t[:, :, -(self.kernel_size - 1) :] if output_final_state else None
            )
            out = F.conv1d(
                x_t,
                self.conv1d.weight,
                self.conv1d.bias,
                padding=self.kernel_size - 1,
                groups=d,
            )[:, :, :t]

        if self.activation == "silu":
            out = F.silu(out)

        out = out.transpose(1, 2)
        return out, new_cache


__all__ = ["ShortConvolution"]
