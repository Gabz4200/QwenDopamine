# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

"""Vectorial Exponential Moving Average with data-dependent decay."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["ValueBaselineEMA"]


class ValueBaselineEMA(nn.Module):
    r"""Vectorial Exponential Moving Average with data-dependent decay.

    Tracks the expectation of a vectorial reward signal (e.g., statistics like
    mean, max, min, std, sum) using an EMA where the decay rate α_t is generated
    from the current input x_t. This is mathematically equivalent to a minLSTM-style
    recurrence operating on a k-dimensional vector.

    For each statistic dimension i ∈ {1..k}:
        α_t[i] = σ(W_α[i] x_t + b_α[i])
        V_t[i] = α_t[i] ⊙ R_t[i] + (1 - α_t[i]) ⊙ V_{t-1}[i]
        A_t[i] = R_t[i] - V_{t-1}[i]

    Args:
        d_model (int): Input feature dimension.
        k_stats (int): Number of reward statistics dimensions.
        init_alpha (float, optional): Initial decay rate before sigmoid.
            Default: 0.1 (slow initial adaptation).

    Shape:
        - x: (B, d_model)
        - R_stats: (B, k_stats)
        - V_prev: (B, k_stats)
        - Returns: V_t (B, k_stats), A_t (B, k_stats)
    """

    def __init__(
        self,
        d_model: int,
        k_stats: int,
        init_alpha: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if k_stats <= 0:
            raise ValueError("k_stats must be positive.")
        if not (0 < init_alpha < 1):
            raise ValueError("init_alpha must be in (0, 1).")

        self.d_model = d_model
        self.k_stats = k_stats

        self.alpha_proj = nn.Linear(d_model, k_stats)

        with torch.no_grad():
            init_bias = torch.log(torch.tensor(init_alpha / (1 - init_alpha)))
            self.alpha_proj.bias.fill_(init_bias)
            self.alpha_proj.weight.zero_()

    def forward(
        self,
        x: Tensor,
        R_stats: Tensor,
        V_prev: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, d_model) - Current input features.
            R_stats: (B, k_stats) - Vectorial reward statistics at step t.
            V_prev: (B, k_stats) - Previous EMA baseline.

        Returns:
            V_t: (B, k_stats) - Updated baseline.
            A_t: (B, k_stats) - Instantaneous advantage vector.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (B, d_model), got {x.shape}.")
        if R_stats.shape != V_prev.shape:
            raise ValueError(
                f"R_stats {R_stats.shape} and V_prev {V_prev.shape} must match."
            )
        if R_stats.size(-1) != self.k_stats:
            raise ValueError(
                f"Expected k_stats={self.k_stats}, got {R_stats.size(-1)}."
            )

        # α_t = σ(W_α x_t + b_α)
        alpha_t = torch.sigmoid(self.alpha_proj(x))  # (B, k_stats)

        # V_t = α_t ⊙ R_t + (1 - α_t) ⊙ V_{t-1}
        V_t = alpha_t * R_stats + (1.0 - alpha_t) * V_prev

        # A_t = R_t - V_{t-1} (instantaneous advantage before update)
        A_t = R_stats - V_prev

        return V_t, A_t

    def extra_repr(self) -> str:
        r"""extra_repr() -> str

        Return a string with the extra representation of the module."""
        return f"d_model={self.d_model}, k_stats={self.k_stats}"
