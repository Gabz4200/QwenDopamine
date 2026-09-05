"""General-purpose reward conditioning building blocks.

Contains scalers, Fourier feature encoders, and FiLM modulation modules
that are not specific to any particular reward pipeline.
"""

import math

import torch
from einops import rearrange
from torch import nn


class LearnableFourierFeatures(nn.Module):
    r"""Learnable Fourier feature mapping followed by an MLP projection."""

    def __init__(
        self,
        pos_dim: int,
        f_dim: int,
        h_dim: int,
        d_dim: int,
        g_dim: int = 1,
        gamma: float = 1.0,
        include_input: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if pos_dim <= 0:
            raise ValueError("pos_dim must be greater than 0.")
        if f_dim <= 0 or f_dim % 2 != 0:
            raise ValueError("f_dim must be greater than 0 and divisible by 2.")
        if h_dim <= 0:
            raise ValueError("h_dim must be greater than 0.")
        if g_dim <= 0:
            raise ValueError("g_dim must be greater than 0.")
        if d_dim <= 0 or d_dim % g_dim != 0:
            raise ValueError("d_dim must be greater than 0 and divisible by g_dim.")
        if gamma <= 0:
            raise ValueError("gamma must be greater than 0.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.pos_dim = pos_dim
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.d_dim = d_dim
        self.g_dim = g_dim
        self.include_input = include_input
        self.dropout = dropout

        self.enc_f_dim = int(f_dim // 2)
        self.dg_dim = int(d_dim // g_dim)
        self.div_term = math.sqrt(f_dim)

        # MLP input dimension depends on whether the raw input is included.
        self.mlp_in_dim = f_dim + pos_dim if include_input else f_dim
        self.out_dim = d_dim

        self.Wr = nn.Parameter(torch.empty(self.enc_f_dim, pos_dim))
        nn.init.normal_(self.Wr, mean=0.0, std=gamma)

        linear1 = nn.Linear(self.mlp_in_dim, h_dim)
        linear2 = nn.Linear(h_dim, self.dg_dim)
        # Best init for GELU MLP: He/Kaiming for hidden, Xavier small for output
        nn.init.kaiming_uniform_(
            linear1.weight, a=0, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(linear1.bias)
        nn.init.xavier_uniform_(linear2.weight, gain=0.5)
        nn.init.zeros_(linear2.bias)

        mlp_layers: list[nn.Module] = [
            linear1,
            nn.GELU(approximate="tanh"),
        ]
        if dropout > 0.0:
            mlp_layers.append(nn.Dropout(p=dropout))
        mlp_layers.append(linear2)

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos (Tensor): Input coordinates of shape ``(B, L, G, pos_dim)``.

        Returns:
            Tensor: Encoded features of shape ``(B, L, d_dim)``.
        """
        # Move inputs to module device/dtype when possible.
        param = next(self.parameters(), None)
        if param is not None:
            pos = pos.to(device=param.device, dtype=param.dtype)

        if pos.dim() != 4:
            raise ValueError(
                f"Expected pos shape (B, L, G, pos_dim), got {tuple(pos.shape)}."
            )

        if pos.size(-1) != self.pos_dim:
            raise ValueError(
                f"Expected pos.size(-1) == {self.pos_dim}, got {pos.size(-1)}."
            )

        if pos.size(2) != self.g_dim:
            raise ValueError(
                f"Expected input group dimension G == g_dim={self.g_dim}, "
                f"got G={pos.size(2)}."
            )

        XWr = torch.matmul(pos, self.Wr.T)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term

        if self.include_input:
            F = torch.cat([pos, F], dim=-1)

        if F.size(-1) != self.mlp_in_dim:
            raise RuntimeError(
                "Internal LearnableFourierFeatures dimension mismatch. "
                f"Expected MLP input dimension {self.mlp_in_dim}, got {F.size(-1)}."
            )

        Y = self.mlp(F)
        return rearrange(Y, "b l g d -> b l (g d)")

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
        return (
            f"pos_dim={self.pos_dim}, "
            f"f_dim={self.f_dim}, "
            f"h_dim={self.h_dim}, "
            f"d_dim={self.d_dim}, "
            f"g_dim={self.g_dim}, "
            f"include_input={self.include_input}, "
            f"mlp_in_dim={self.mlp_in_dim}, "
            f"out_dim={self.out_dim}"
        )
