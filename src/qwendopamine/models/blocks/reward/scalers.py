"""General-purpose reward conditioning building blocks.

Contains scalers, Fourier feature encoders, and FiLM modulation modules
that are not specific to any particular reward pipeline.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class AsinhScaler(nn.Module):
    r"""Element-wise inverse hyperbolic sine scaling."""

    def __init__(
        self,
        dim: int,
        init_scale: float = 0.1,
        shared_alpha: bool = True,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("AsinhScaler dim must be greater than 0.")
        if init_scale <= 0:
            raise ValueError("AsinhScaler init_scale must be greater than 0.")

        self.dim = dim
        self.init_scale = init_scale
        self.shared_alpha = shared_alpha

        raw_shape = () if shared_alpha else (dim,)
        raw_init = self._inverse_softplus(init_scale)

        # alpha = softplus(raw_alpha) guarantees alpha > 0.
        self.raw_alpha = nn.Parameter(torch.full(raw_shape, raw_init))

    @staticmethod
    def _inverse_softplus(x: float) -> float:
        """Return y such that F.softplus(y) == x."""
        if x > 20.0:
            return x
        return math.log(math.expm1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input by learned asinh factor."""
        if x.size(-1) != self.dim:
            raise ValueError(
                f"AsinhScaler expected last dimension {self.dim}, got {x.size(-1)}."
            )

        input_dtype = x.dtype

        # Compute in float32 for numerical stability.
        x = x.float()
        alpha = F.softplus(self.raw_alpha).float()

        x = torch.asinh(alpha * x)
        return x.to(input_dtype)

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
            f"dim={self.dim}, "
            f"init_scale={self.init_scale}, "
            f"shared_alpha={self.shared_alpha}"
        )


class LearnableSoftsign(nn.Module):
    r"""Learnable Softsign normalization mapping inputs to (-1, 1)."""

    def __init__(
        self,
        per_channel: bool = False,
        num_channels: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if per_channel:
            if num_channels is None:
                raise ValueError(
                    "num_channels must be specified when per_channel=True."
                )
            if num_channels <= 0:
                raise ValueError("num_channels must be greater than 0.")
        if eps <= 0:
            raise ValueError("eps must be greater than 0.")

        self.per_channel = per_channel
        self.num_channels = num_channels
        self.eps = eps

        if per_channel:
            assert num_channels is not None
            shape = (num_channels,)
        else:
            shape = ()
        self.gamma = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor | float) -> torch.Tensor:
        """
        Args:
            x: Input tensor of any shape, or a Python scalar.

        Returns:
            Tensor: Normalized tensor in range (-1, 1) with same shape as input.
        """
        # Convert Python scalars to tensors on the correct device/dtype.
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.gamma.device)

        input_dtype = x.dtype

        # Compute in float32 for stability in bf16/fp16 (eps would be lost).
        x_f = x.float()
        alpha = torch.exp(self.gamma).float()

        out = x_f / (torch.abs(x_f) + alpha + self.eps)
        return out.to(input_dtype)

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
            f"per_channel={self.per_channel}, "
            f"num_channels={self.num_channels}, "
            f"eps={self.eps}"
        )
