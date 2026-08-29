"""Reward-specific extractor and FiLM modules.

These modules depend on the general-purpose building blocks in
:mod:`~qwendopamine.models.blocks.reward.components`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from qwendopamine.models.blocks.reward.components import (
    AsinhScaler,
    LearnableFourierFeatures,
    TokenWiseFiLM,
)


class RewardStatisticsExtractor(nn.Module):
    r"""Extracts statistical features from reward tensors.

    Normalizes reward tensors of various shapes to a common layout
    broadcastable to ``(B, L, K)``, then computes six statistics per
    position: median, mean, max, min, standard deviation, and sum.

    Args:
        normalize (bool, optional): If ``True``, applies :class:`AsinhScaler`
            to the six statistics. Default: ``True``.
        reward_init_scale (float, optional): Initial positive scaling factor
            used by :class:`AsinhScaler` when ``normalize=True``.
            Default: ``0.1``.
        shared_alpha (bool, optional): If ``True``, uses one shared scaling
            factor for all six statistics in :class:`AsinhScaler`. Default: ``True``.

    Shape:
        - reward_values: Scalar, 1D, 2D, or 3D tensor. Common shapes include
          ``(B, L, K)``, ``(B, L)``, ``(B, K)``, ``(L, K)``, ``(K,)``,
          ``(L,)``, or scalar.
        - Output: ``(B, L, 6)`` where the last dimension contains
          ``[median, mean, max, min, std, sum]`` in that order.

    Examples::

        >>> extractor = RewardStatisticsExtractor()
        >>> rewards = torch.randn(2, 5, 10)
        >>> stats = extractor(rewards, batch_size=2, seq_len=5)
        >>> stats.shape
        torch.Size([2, 5, 6])
    """

    def __init__(
        self,
        normalize: bool = True,
        reward_init_scale: float = 0.1,
        shared_alpha: bool = True,
        reward_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if reward_init_scale <= 0:
            raise ValueError("reward_init_scale must be greater than 0.")
        if not (0.0 <= reward_dropout < 1.0):
            raise ValueError("reward_dropout must be in [0.0, 1.0).")

        self.normalize = normalize
        self.reward_init_scale = reward_init_scale
        self.shared_alpha = shared_alpha
        self.reward_dropout = reward_dropout

        self.scaler: nn.Module = (
            AsinhScaler(
                dim=6,
                init_scale=reward_init_scale,
                shared_alpha=shared_alpha,
            )
            if normalize
            else nn.Identity()
        )

    @staticmethod
    def _normalize_reward_values(
        reward_values: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Normalize reward_values to a tensor broadcastable to (B, L, K)."""
        ndim = reward_values.dim()
        dispatch = {
            0: lambda: reward_values.view(1, 1, 1),
            1: RewardStatisticsExtractor._normalize_1d,
            2: RewardStatisticsExtractor._normalize_2d,
            3: RewardStatisticsExtractor._normalize_3d,
        }
        if ndim not in dispatch:
            raise ValueError(
                "reward_values must be a scalar, 1D, 2D, or 3D tensor. "
                f"Got shape {tuple(reward_values.shape)}."
            )
        return dispatch[ndim](reward_values, batch_size=batch_size, seq_len=seq_len)

    @staticmethod
    def _normalize_1d(
        reward_values: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        # (L,) -> (1, L, 1)
        # (B,) -> (B, 1, 1), when L == 1
        # (K,) -> (1, 1, K)
        length = reward_values.size(0)
        if length == seq_len:
            return reward_values.view(1, seq_len, 1)
        if seq_len == 1 and length == batch_size:
            return reward_values.view(batch_size, 1, 1)
        return reward_values.view(1, 1, -1)

    @staticmethod
    def _normalize_2d(
        reward_values: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        # (B, L) -> (B, L, 1)
        # (B, K) -> (B, 1, K)
        # (L, K) -> (1, L, K)
        # (1, L) -> (1, L, 1)
        # (1, K) -> (1, 1, K)
        first, second = reward_values.shape
        if (first, second) == (batch_size, seq_len):
            return reward_values.unsqueeze(-1)
        if first == batch_size:
            return reward_values.unsqueeze(1)
        if first == seq_len:
            return reward_values.unsqueeze(0)
        if first == 1:
            if second == seq_len:
                return reward_values.unsqueeze(-1)
            return reward_values.unsqueeze(1)
        raise ValueError(
            "Could not normalize 2D reward_values. "
            f"Got reward_values.shape={tuple(reward_values.shape)}, "
            f"batch_size={batch_size}, seq_len={seq_len}. "
            "Expected one of: (B, L), (B, K), (L, K), (1, L), or (1, K)."
        )

    @staticmethod
    def _normalize_3d(
        reward_values: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        # Already 3D. Allow broadcastable batch and sequence dimensions.
        if reward_values.size(0) not in (1, batch_size):
            raise ValueError(
                "3D reward_values batch dimension must be either 1 or batch_size. "
                f"Got reward_values.shape={tuple(reward_values.shape)}, "
                f"batch_size={batch_size}."
            )
        if reward_values.size(1) not in (1, seq_len):
            raise ValueError(
                "3D reward_values sequence dimension must be either 1 or seq_len. "
                f"Got reward_values.shape={tuple(reward_values.shape)}, "
                f"seq_len={seq_len}."
            )
        return reward_values

    def forward(
        self, reward_values: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """
        Args:
            reward_values (Tensor): Reward tensor of various shapes.
            batch_size (int): Batch dimension size.
            seq_len (int): Sequence length dimension size.

        Returns:
            Tensor: Statistics tensor of shape ``(B, L, 6)``.
        """
        # Move inputs to module device/dtype when possible.
        param = next(self.parameters(), None)
        target_dtype = param.dtype if param is not None else reward_values.dtype
        if param is not None:
            reward_values = reward_values.to(device=param.device, dtype=param.dtype)

        reward_values = self._normalize_reward_values(
            reward_values, batch_size=batch_size, seq_len=seq_len
        )

        if reward_values.size(-1) == 0:
            raise ValueError("reward_values must contain at least one reward channel.")

        if self.training and self.reward_dropout > 0.0:
            reward_values = F.dropout(
                reward_values, p=self.reward_dropout, training=True
            )

        # Compute statistics in float32 for numerical stability.
        reward_values = reward_values.float()

        reward_median = reward_values.median(dim=-1, keepdim=True)[0]
        reward_mean = reward_values.mean(dim=-1, keepdim=True)
        reward_max = reward_values.max(dim=-1, keepdim=True)[0]
        reward_min = reward_values.min(dim=-1, keepdim=True)[0]

        # Population standard deviation avoids NaN when K == 1.
        reward_std = reward_values.std(dim=-1, keepdim=True, correction=0)

        # Sum of all rewards.
        reward_sum = reward_values.sum(dim=-1, keepdim=True)

        reward_stats = torch.cat(
            [
                reward_median,
                reward_mean,
                reward_max,
                reward_min,
                reward_std,
                reward_sum,
            ],
            dim=-1,
        )

        reward_stats = self.scaler(reward_stats)

        # Convert back to target dtype if specified
        if target_dtype is not None:
            reward_stats = reward_stats.to(target_dtype)

        return reward_stats

    def extra_repr(self) -> str:
        return (
            f"normalize={self.normalize}, "
            f"reward_init_scale={self.reward_init_scale}, "
            f"shared_alpha={self.shared_alpha}, "
            f"reward_dropout={self.reward_dropout}"
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

        return cond

    def extra_repr(self) -> str:
        return (
            f"f_dim={self.f_dim}, "
            f"h_dim={self.h_dim}, "
            f"d_dim={self.d_dim}, "
            f"g_dim={self.g_dim}, "
            f"include_input={self.include_input}, "
            f"dropout={self.dropout}"
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

        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}, dropout={self.dropout}"
