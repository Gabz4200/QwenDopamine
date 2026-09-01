r"""Normalization layers and token mask utilities for Qwen-style architectures."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization without mean centering."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def reset_parameters(self) -> None:
        r"""Reset the learned scale weight back to a vector of ones."""
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Apply RMSNorm to hidden states."""
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_normed = hidden_states_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight.to(input_dtype) * hidden_states_normed.to(input_dtype)


class RMSNormGated(nn.Module):
    r"""RMSNorm with optional element-wise activation gating."""

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, *, activation: str = "silu"
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(
        self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Apply gated RMSNorm to hidden states."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * torch.nn.functional.silu(
                gate.to(torch.float32)
            )
        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Zero out hidden states at padded token positions."""
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states
