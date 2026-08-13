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
