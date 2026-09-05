"""General-purpose reward conditioning building blocks.

Contains scalers, Fourier feature encoders, and FiLM modulation modules
that are not specific to any particular reward pipeline.
"""

import torch
import torch.nn.functional as F
from torch import nn


class TokenWiseFiLM(nn.Module):
    r"""Token-wise Feature-wise Linear Modulation."""

    def __init__(
        self,
        dim: int,
        cond_dim: int | None = None,
        identity_init: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("TokenWiseFiLM dim must be greater than 0.")

        self.dim = dim
        self.cond_dim = dim if cond_dim is None else cond_dim
        self.dropout = dropout

        if self.cond_dim <= 0:
            raise ValueError("TokenWiseFiLM cond_dim must be greater than 0.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.gamma_proj = nn.Linear(self.cond_dim, dim)
        self.beta_proj = nn.Linear(self.cond_dim, dim)

        if identity_init:
            # Initial behavior: y = x * 1 + 0.
            nn.init.zeros_(self.gamma_proj.weight)
            nn.init.ones_(self.gamma_proj.bias)

            nn.init.zeros_(self.beta_proj.weight)
            nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape ``(B, D)`` or ``(B, L, D)``.
            cond (Tensor): Conditioning tensor.

        Returns:
            Tensor: Modulated tensor with the same shape as ``x``.
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"TokenWiseFiLM expects x with shape (B, D) or (B, L, D), "
                f"got {tuple(x.shape)}."
            )

        cond = self._broadcast_cond(x, cond)

        if self.training and self.dropout > 0.0:
            cond = F.dropout(cond, p=self.dropout, training=True)

        # Primary contract: conditioning has the expected feature dimension.
        if cond.size(-1) == self.cond_dim:
            gamma = self.gamma_proj(cond)
            beta = self.beta_proj(cond)

        # Backward-compatible contract: concatenated [gamma, beta] conditioning.
        elif self.cond_dim == self.dim and cond.size(-1) == 2 * self.dim:
            gamma_in, beta_in = torch.chunk(cond, chunks=2, dim=-1)
            gamma = self.gamma_proj(gamma_in)
            beta = self.beta_proj(beta_in)

        else:
            raise ValueError(
                "TokenWiseFiLM conditioning dimension mismatch. "
                f"Expected cond.shape[-1] == {self.cond_dim} "
                f"or backward-compatible {2 * self.dim}, got {cond.size(-1)}."
            )

        result: torch.Tensor = x * gamma + beta
        return result

    def _broadcast_cond(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Align ``cond`` to the same rank as ``x`` for broadcasting."""
        if cond.dim() == 0:
            cond = cond.unsqueeze(0)

        if cond.dim() == 1:
            cond = cond.unsqueeze(0)

        # Align conditioning rank with input rank for broadcasting.
        if x.dim() == 3 and cond.dim() == 2:
            cond = cond.unsqueeze(1)
        elif x.dim() == 2 and cond.dim() == 3:
            if cond.size(1) != 1:
                raise ValueError(
                    "When x has shape (B, D), cond with shape (B, L, C) is only "
                    f"valid if L == 1. Got cond.shape={tuple(cond.shape)}."
                )
            cond = cond.squeeze(1)
        elif x.dim() != cond.dim():
            raise ValueError(
                "Unsupported combination of x and cond shapes: "
                f"x={tuple(x.shape)}, cond={tuple(cond.shape)}."
            )

        return cond

    def extra_repr(self) -> str:
        r"""extra_repr() -> str

        Return a string with the extra representation of the module."""
        return f"dim={self.dim}, cond_dim={self.cond_dim}, dropout={self.dropout}"
