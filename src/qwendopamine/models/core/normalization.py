r"""Normalization layers and token mask utilities for Qwen-style architectures."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    r"""RMSNorm(hidden_size: int, eps: float = 1e-6) -> None

    Root Mean Square Layer Normalization without mean centering.

    Computes ``x / sqrt(mean(x^2) + eps) * weight``.

    Args:
        hidden_size (int): Dimension of the input to normalize.
        eps (float): Epsilon added to the variance denominator for
            numerical stability. Default: ``1e-6``.

    Examples::
        >>> norm = RMSNorm(hidden_size=512)
        >>> out = norm(torch.randn(2, 4, 512))
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def reset_parameters(self) -> None:
        r"""reset_parameters() -> None

        Reset the learned scale weight to a vector of ones.

        Returns:
            None
        """
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Apply RMSNorm to hidden states.

        Args:
            hidden_states (torch.Tensor): Input tensor ``[..., hidden_size]``.

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_normed = hidden_states_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight.to(input_dtype) * hidden_states_normed.to(input_dtype)


class RMSNormGated(nn.Module):
    r"""RMSNormGated(hidden_size: int, eps: float = 1e-6, *, activation: str = "silu") -> None

    RMSNorm with element-wise activation gating.

    Computes ``x / sqrt(mean(x^2) + eps) * weight * silu(gate)``.

    Args:
        hidden_size (int): Dimension of the input to normalize.
        eps (float): Epsilon added to the variance denominator. Default: ``1e-6``.
        activation (str): Activation for the gate. Default: ``"silu"``.

    Examples::
        >>> norm = RMSNormGated(hidden_size=512)
        >>> out = norm(torch.randn(2, 4, 512), gate=torch.randn(2, 4, 512))
    """

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
        r"""forward(hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor

        Apply gated RMSNorm to hidden states.

        Args:
            hidden_states (torch.Tensor): Input tensor ``[..., hidden_size]``.
            gate (torch.Tensor | None): Gate tensor matching the shape of
                ``hidden_states``. When ``None``, no gating is applied.

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
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
    r"""apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor

    Zero out hidden states at padded token positions.

    Args:
        hidden_states (torch.Tensor): Input tensor ``[B, T, D]``.
        attention_mask (torch.Tensor | None): Boolean mask ``[B, T]`` where
            ``True`` means a real token. When ``None``, returns the input
            unchanged.

    Returns:
        torch.Tensor: Masked tensor of the same shape as ``hidden_states``.
    """
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states
