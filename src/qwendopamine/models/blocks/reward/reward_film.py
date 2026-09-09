"""Reward FiLM modulation wrapper around :class:`TokenWiseFiLM`."""

import torch
from torch import nn

from qwendopamine.models.blocks.reward.components import (
    TokenWiseFiLM,
)


class RewardFiLM(nn.Module):
    r"""Token-wise FiLM modulation conditioned on reward encoding.

    Projects input features to ``hidden_dim`` (if needed), then applies
    Feature-wise Linear Modulation using a conditioning tensor.

    Args:
        dim (int): Input feature dimension.
        hidden_dim (int): Hidden feature dimension after projection and conditioning.
        identity_init (bool, optional): If ``True``, initializes FiLM as approximately
            identity: ``gamma = 1``, ``beta = 0``. Default: ``True``.
        dropout (float, optional): Dropout probability applied to the conditioning tensor.
            Default: ``0.0``.

    Shape:
        - x: ``(D)``, ``(B, D)``, or ``(B, L, D)``
        - cond: ``(B, L, hidden_dim)`` or broadcastable
        - Output: Same leading shape as ``x`` with feature dimension ``hidden_dim``.

    Examples::

        >>> film = RewardFiLM(dim=32, hidden_dim=64)
        >>> x = torch.randn(2, 5, 32)
        >>> cond = torch.randn(2, 5, 64)
        >>> out = film(x, cond)
        >>> out.shape
        torch.Size([2, 5, 64])
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        identity_init: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be greater than 0.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.x_proj: nn.Module = (
            nn.Linear(dim, hidden_dim) if dim != hidden_dim else nn.Identity()
        )
        if isinstance(self.x_proj, nn.Linear):
            # Xavier small-gain keeps residual stable when dims differ
            nn.init.xavier_uniform_(self.x_proj.weight, gain=0.5)
            if self.x_proj.bias is not None:
                nn.init.zeros_(self.x_proj.bias)

        self.film = TokenWiseFiLM(
            dim=hidden_dim,
            cond_dim=hidden_dim,
            identity_init=identity_init,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input feature tensor of shape ``(D)``, ``(B, D)``, or ``(B, L, D)``.
            cond (Tensor): Conditioning tensor of shape ``(B, L, hidden_dim)``, ``(B, hidden_dim)``,
                ``(hidden_dim,)``, or broadcastable.

        Returns:
            Tensor: Modulated tensor with same leading shape as ``x`` and feature dim ``hidden_dim``.
        """
        # Move inputs to module device/dtype when possible.
        param = next(self.parameters(), None)
        if param is not None:
            x = x.to(device=param.device, dtype=param.dtype)
            cond = cond.to(device=param.device, dtype=param.dtype)

        orig_x_dim = x.dim()

        if orig_x_dim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif orig_x_dim == 2:
            x = x.unsqueeze(1)
        elif orig_x_dim != 3:
            raise ValueError(
                f"Expected x with shape (D,), (B, D), or (B, L, D), got {tuple(x.shape)}."
            )

        x_hidden = self.x_proj(x)

        # ``TokenWiseFiLM`` aligns any broadcastable cond rank internally
        # via ``broadcast_cond``; no pre-expansion is needed here.
        output = self.film(x_hidden, cond)

        if orig_x_dim == 1:
            output = output.squeeze(0).squeeze(0)
        elif orig_x_dim == 2:
            output = output.squeeze(1)

        result: torch.Tensor = output
        return result

    def extra_repr(self) -> str:
        r"""extra_repr() -> str

        Return a string with the extra representation of the module."""
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}, dropout={self.dropout}"
