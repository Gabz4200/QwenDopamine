"""Reward-specific extractor and FiLM modules.

These modules depend on the general-purpose building blocks in
:mod:`~qwendopamine.models.blocks.reward.components`.
"""

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

        batch_size, seq_len, _ = x.shape

        # Align conditioning tensor to (B, L, hidden_dim) if needed.
        # Handle 1D cond (hidden_dim,) -> (1, 1, hidden_dim) -> broadcast to (B, L, hidden_dim)
        if cond.dim() == 1:
            if cond.size(0) == self.hidden_dim:
                cond = cond.view(1, 1, self.hidden_dim).expand(batch_size, seq_len, -1)
            else:
                raise ValueError(
                    f"1D cond feature dimension {cond.size(0)} must be {self.hidden_dim}."
                )
        # Handle 2D cond (B, hidden_dim) or (L, hidden_dim) or (1, hidden_dim)
        elif cond.dim() == 2:
            if cond.size(0) == batch_size and cond.size(1) == self.hidden_dim:
                # (B, hidden_dim) -> (B, 1, hidden_dim) -> broadcast to (B, L, hidden_dim)
                cond = cond.unsqueeze(1).expand(-1, seq_len, -1)
            elif cond.size(0) == seq_len and cond.size(1) == self.hidden_dim:
                # (L, hidden_dim) -> (1, L, hidden_dim) -> broadcast to (B, L, hidden_dim)
                cond = cond.unsqueeze(0).expand(batch_size, -1, -1)
            elif cond.size(0) == 1 and cond.size(1) == self.hidden_dim:
                # (1, hidden_dim) -> (1, 1, hidden_dim) -> broadcast to (B, L, hidden_dim)
                cond = cond.unsqueeze(1).expand(batch_size, seq_len, -1)
            else:
                raise ValueError(
                    f"Cannot broadcast cond shape {tuple(cond.shape)} to (B, L, {self.hidden_dim})."
                )
        elif cond.dim() == 3:
            if cond.size(0) not in (1, batch_size):
                raise ValueError(
                    f"cond batch dimension {cond.size(0)} must be 1 or {batch_size}."
                )
            if cond.size(1) not in (1, seq_len):
                raise ValueError(
                    f"cond sequence dimension {cond.size(1)} must be 1 or {seq_len}."
                )
            if cond.size(2) != self.hidden_dim:
                raise ValueError(
                    f"cond feature dimension {cond.size(2)} must be {self.hidden_dim}."
                )
            # Broadcast if needed
            if cond.size(0) == 1:
                cond = cond.expand(batch_size, -1, -1)
            if cond.size(1) == 1:
                cond = cond.expand(-1, seq_len, -1)
        else:
            raise ValueError(f"cond must be 1D, 2D, or 3D tensor, got {cond.dim()}D.")

        x_hidden = self.x_proj(x)

        if cond.shape[-1] != x_hidden.shape[-1]:
            raise RuntimeError(
                "Conditioning and input feature dimensions must match before TokenWiseFiLM. "
                f"Got cond.shape[-1]={cond.shape[-1]}, "
                f"x_hidden.shape[-1]={x_hidden.shape[-1]}."
            )

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
