"""Reward-specific extractor and FiLM modules.

These modules depend on the general-purpose building blocks in
:mod:`~qwendopamine.models.blocks.reward.components`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from qwendopamine.models.blocks.reward.components import (
    AsinhScaler,
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
            1: RewardStatisticsExtractor._normalize_1d,
            2: RewardStatisticsExtractor._normalize_2d,
            3: RewardStatisticsExtractor._normalize_3d,
        }
        if ndim not in dispatch:
            raise ValueError(
                "reward_values must be a scalar, 1D, 2D, or 3D tensor. "
                f"Got shape {tuple(reward_values.shape)}."
            )
        if ndim == 0:
            return reward_values.view(1, 1, 1)
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

        result: torch.Tensor = reward_stats
        return result

    def extra_repr(self) -> str:
        r"""extra_repr() -> str

        Return a string with the extra representation of the module."""
        return (
            f"normalize={self.normalize}, "
            f"reward_init_scale={self.reward_init_scale}, "
            f"shared_alpha={self.shared_alpha}, "
            f"reward_dropout={self.reward_dropout}"
        )
