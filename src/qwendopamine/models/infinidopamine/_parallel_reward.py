"""Parallel reward branch assembly for InfiniDopamineDecoderLayer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from qwendopamine.models.infinidopamine._gated_reward_net import (
    InfiniDopamineGatedRewardNet,
)
from qwendopamine.models.infinidopamine._norm import InfiniDopamineRMSNorm

if TYPE_CHECKING:
    from qwendopamine.models.infinidopamine.configs import InfiniDopamineTextConfig


class ParallelRewardBranch(nn.Module):
    """Data-dependent parallel reward branch with its own normalization."""

    def __init__(self, config: InfiniDopamineTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.reward_branch = InfiniDopamineGatedRewardNet(config, layer_idx)
        self.reward_branch_norm = InfiniDopamineRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.reward_gate_proj = nn.Linear(config.hidden_size, 1, bias=True)
        nn.init.zeros_(self.reward_gate_proj.weight)
        nn.init.constant_(
            self.reward_gate_proj.bias,
            getattr(config, "reward_gate_init_bias", -5.0),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        reward_values: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        reward_out = self.reward_branch(
            hidden_states=hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            reward_values=reward_values,
            **kwargs,
        )
        reward_out = self.reward_branch_norm(reward_out)
        gate = torch.sigmoid(self.reward_gate_proj(hidden_states))
        return gate * reward_out  # type: ignore[no-any-return-implicit]


def build_parallel_reward_branch(
    config: InfiniDopamineTextConfig,
    layer_idx: int,
) -> ParallelRewardBranch:
    """Build a parallel reward branch with near-zero initial gating."""
    return ParallelRewardBranch(config, layer_idx)
