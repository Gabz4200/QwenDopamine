r"""Normalization layers and token mask utilities for Qwen-style architectures."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    r"""RMSNorm(hidden_size, eps=1e-6)

    Applies Root Mean Square Layer Normalization over the last dimension without feature mean centering.

    .. math::
        \text{RMSNorm}(x) = w \odot \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}

    Args:
        hidden_size (int): Hidden dimension size of the input tensor.
        eps (float, optional): Epsilon value added to the variance calculation for numerical stability.
            Default: ``1e-6``.

    Examples::

        >>> norm = RMSNorm(hidden_size=64)
        >>> x = torch.randn(2, 5, 64)
        >>> out = norm(x)
        >>> out.shape
        torch.Size([2, 5, 64])
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def reset_parameters(self) -> None:
        r"""Reset the learned scale weight back to a vector of ones."""
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""forward(hidden_states) -> Tensor

        Args:
            hidden_states (Tensor): Input feature tensor of shape :math:`(..., \text{hidden\_size})`.

        Returns:
            Tensor: Normalized feature tensor of same shape and dtype as ``hidden_states``.
        """
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_normed = hidden_states_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight.to(input_dtype) * hidden_states_normed.to(input_dtype)


class RMSNormGated(nn.Module):
    r"""RMSNormGated(hidden_size, eps=1e-6, *, activation="silu")

    Applies RMSNorm with optional element-wise activation gating on normalized hidden states.

    .. math::
        y = (w \odot \text{RMSNorm}(x)) \odot \text{SiLU}(g)

    Args:
        hidden_size (int): Hidden dimension size of input feature tensor.
        eps (float, optional): Small constant added for numerical stability during normalization.
            Default: ``1e-6``.
        activation (str, optional): Activation function applied to the gating tensor. Default: ``"silu"``.

    Examples::

        >>> norm_gated = RMSNormGated(hidden_size=64)
        >>> x = torch.randn(2, 5, 64)
        >>> gate = torch.randn(2, 5, 64)
        >>> out = norm_gated(x, gate)
        >>> out.shape
        torch.Size([2, 5, 64])
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
        r"""forward(hidden_states, gate=None) -> Tensor

        Args:
            hidden_states (Tensor): Input feature tensor of shape :math:`(..., \text{hidden\_size})`.
            gate (Tensor, optional): Optional gating tensor of same shape as ``hidden_states``. Default: ``None``.

        Returns:
            Tensor: Gated normalized hidden states of same shape as ``hidden_states``.
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
    r"""apply_mask_to_padding_states(hidden_states, attention_mask=None) -> Tensor

    Zeros out hidden state vectors corresponding to padded token positions before recurrent scans.

    Args:
        hidden_states (Tensor): Sequence hidden states of shape :math:`(B, L, D)`.
        attention_mask (Tensor, optional): Binary attention mask tensor of shape :math:`(B, L)` where ``1``
            indicates valid tokens and ``0`` indicates padding tokens. Default: ``None``.

    Returns:
        Tensor: Masked sequence hidden states of shape :math:`(B, L, D)`.
    """
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states
