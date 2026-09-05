"""MLP block: SwiGLU activation and LLaMA-style MLP wrapper.

Moved from ``model.py`` for size.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig


class SwiGLU(nn.Module):
    r"""SwiGLU activation with parallel-gated linear projections.

    Computes ``w3(silu(w1(x)) * w2(x))``.

    Args:
        in_features (int): Input dimension.
        hidden_features (int): Hidden (intermediate) dimension.
        bias (bool): Whether to include bias on linear layers. Default: ``False``.
    """

    def __init__(
        self, in_features: int, hidden_features: int, bias: bool = False
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x: torch.Tensor) -> torch.Tensor

        Args:
            x (torch.Tensor): Input ``[..., in_features]``.

        Returns:
            torch.Tensor: Activated output ``[..., in_features]``.
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = F.silu(x1) * x2
        x = self.w3(x)
        result: torch.Tensor = x
        return result


class LLaMAMLP(nn.Module):
    r"""LLaMA-style MLP wrapping a SwiGLU activation.

    Args:
        config (GDN2GPTConfig): Configuration with ``n_embd``,
            ``intermediate_size``, and ``bias``.
    """

    def __init__(self, config: GDN2GPTConfig) -> None:
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd, config.intermediate_size, bias=config.bias)
        # Dropout optional; default 0.0 in the original config.
        self.dropout = config.mlp_dropout if hasattr(config, "mlp_dropout") else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x: torch.Tensor) -> torch.Tensor

        Args:
            x (torch.Tensor): Input ``[..., n_embd]``.

        Returns:
            torch.Tensor: Output ``[..., n_embd]``.
        """
        x = self.swiglu(x)
        if self.training and self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=True)
        result: torch.Tensor = x
        return result
