from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Applies RMS normalization over the last dimension without centering,
    followed by a learned weight scale.

    .. math::
        \text{RMSNorm}(x) = w \odot \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}

    Args:
        hidden_size (int): last dimension of the input tensor.
        eps (float): epsilon added to variance for numerical stability. Default: ``1e-6``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class RMSNormGated(nn.Module):
    r"""Reference Qwen3-style gated RMSNorm.

    This follows the upstream Qwen3.5 implementation where the gate is applied to
    the normalized hidden states after RMS scaling.
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
    """Zero out padded tokens before recurrent computation."""
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states
