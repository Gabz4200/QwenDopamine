"""Reward-specific extractor and FiLM modules.

These modules depend on the general-purpose building blocks in
:mod:`~qwendopamine.models.blocks.reward.components`.
"""

import torch
from torch import nn

from qwendopamine.models.blocks.reward.components import (
    LearnableFourierFeatures,
)


class RewardFourierEncoder(nn.Module):
    r"""Encodes reward statistics with learnable Fourier features and MLP.

    Takes 6-dimensional reward statistics (median, mean, max, min, std, sum),
    applies learnable Fourier feature mapping followed by an MLP projection
    to produce a conditioning tensor of dimension ``d_dim``.

    Args:
        f_dim (int, optional): Number of Fourier feature channels. Must be
            divisible by 2. Default: ``32``.
        h_dim (int, optional): Hidden dimension of the MLP. Default: ``64``.
        d_dim (int, optional): Output feature dimension. Must be divisible
            by ``g_dim``. Default: ``64``.
        g_dim (int, optional): Group dimension used in the output rearrangement.
            Default: ``1``.
        gamma (float, optional): Standard deviation for Fourier projection init.
            Default: ``1.0``.
        include_input (bool, optional): If ``True``, concatenates the raw 6-dim
            statistics to the Fourier features before the MLP. Default: ``True``.
        dropout (float, optional): Dropout probability applied in the MLP.
            Default: ``0.0``.

    Shape:
        - Input: ``(B, L, 6)``
        - Output: ``(B, L, d_dim)``

    Examples::

        >>> encoder = RewardFourierEncoder(f_dim=32, h_dim=64, d_dim=64)
        >>> stats = torch.randn(2, 5, 6)
        >>> cond = encoder(stats)
        >>> cond.shape
        torch.Size([2, 5, 64])
    """

    def __init__(
        self,
        f_dim: int = 32,
        h_dim: int = 64,
        d_dim: int = 64,
        g_dim: int = 1,
        gamma: float = 1.0,
        include_input: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

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

        self.f_dim = f_dim
        self.h_dim = h_dim
        self.d_dim = d_dim
        self.g_dim = g_dim
        self.include_input = include_input
        self.dropout = dropout

        self.fourier = LearnableFourierFeatures(
            pos_dim=6,
            f_dim=f_dim,
            h_dim=h_dim,
            d_dim=d_dim,
            g_dim=g_dim,
            gamma=gamma,
            include_input=include_input,
            dropout=dropout,
        )

    def forward(self, reward_stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reward_stats (Tensor): Statistics tensor of shape ``(B, L, 6)``.

        Returns:
            Tensor: Conditioning tensor of shape ``(B, L, d_dim)``.
        """
        if reward_stats.dim() != 3:
            raise ValueError(
                f"Expected reward_stats shape (B, L, 6), got {tuple(reward_stats.shape)}."
            )

        if reward_stats.size(-1) != 6:
            raise ValueError(
                f"Expected reward_stats.size(-1) == 6, got {reward_stats.size(-1)}."
            )

        # LearnableFourierFeatures expects (B, L, G, pos_dim), with G == g_dim.
        cond = self.fourier(reward_stats.unsqueeze(2))

        result: torch.Tensor = cond
        return result

    def extra_repr(self) -> str:
        r"""extra_repr() -> str

        Return a string with the extra representation of the module."""
        return (
            f"f_dim={self.f_dim}, "
            f"h_dim={self.h_dim}, "
            f"d_dim={self.d_dim}, "
            f"g_dim={self.g_dim}, "
            f"include_input={self.include_input}, "
            f"dropout={self.dropout}"
        )
