"""InfiniDopamine decoder block.

The class definitions for the per-block components live in dedicated
sibling modules:

  - :mod:`._gated_delta_net`  — :class:`InfiniDopamineGatedDeltaNet`
  - :mod:`._gated_reward_net` — :class:`InfiniDopamineGatedRewardNet`
  - :mod:`._attention`        — :class:`InfiniDopamineAttention`
  - :mod:`._mlp`              — :class:`InfiniDopamineMLP`
  - :mod:`._norm`             — :class:`InfiniDopamineRMSNorm`

This module owns :class:`InfiniDopamineDecoderLayer`, which composes the
above components based on ``config.layer_types[layer_idx]``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextSparseMoeBlock,
)

from qwendopamine.models.infinidopamine._attention import InfiniDopamineAttention
from qwendopamine.models.infinidopamine._gated_delta_net import (
    InfiniDopamineGatedDeltaNet,
)
from qwendopamine.models.infinidopamine._gated_reward_net import (
    InfiniDopamineGatedRewardNet,
)
from qwendopamine.models.infinidopamine._layer_factory import (
    resolve_block_type,
    should_enable_parallel_reward,
    should_use_sparse_moe,
)
from qwendopamine.models.infinidopamine._mlp import InfiniDopamineMLP
from qwendopamine.models.infinidopamine._norm import InfiniDopamineRMSNorm
from qwendopamine.models.infinidopamine._parallel_reward import (
    build_parallel_reward_branch,
)
from qwendopamine.models.infinidopamine.configs import InfiniDopamineTextConfig


class InfiniDopamineDecoderLayer(GradientCheckpointingLayer):
    r"""InfiniDopamine decoder block.

    The main mixer is selected explicitly by ``config.layer_types[layer_idx]``.
    No implicit replacement of GDN-2 with GatedRewardNet happens based on the
    next layer's type: GatedRewardNet is opt-in via
    ``config.parallel_reward_layers`` and runs as a parallel branch on top of
    whichever main mixer was chosen.

    Layer configuration table:

    =====================  ==========================================
    block_type             main mixer
    =====================  ==========================================
    linear_attention/gdn2  :class:`InfiniDopamineGatedDeltaNet`
    gated_reward_net/reinforced_delta  :class:`InfiniDopamineGatedRewardNet`
    full_attention/sliding_attention  :class:`InfiniDopamineAttention`
    =====================  ==========================================
    """

    def __init__(self, config: InfiniDopamineTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_dropout = getattr(
            config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
        )
        self.block_type = resolve_block_type(config, layer_idx)
        family = self.block_type
        if family == "linear":
            self.linear_attn = InfiniDopamineGatedDeltaNet(config, layer_idx)
        elif family == "attention":
            self.self_attn = InfiniDopamineAttention(config, layer_idx)
        elif family == "reward":
            self.linear_attn = InfiniDopamineGatedRewardNet(config, layer_idx)

        if should_enable_parallel_reward(config, layer_idx):
            self.reward_branch = build_parallel_reward_branch(config, layer_idx)

        if should_use_sparse_moe(config, layer_idx):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = InfiniDopamineMLP(config, config.intermediate_size)
        self.input_layernorm = InfiniDopamineRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = InfiniDopamineRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @property
    def reward_gate_proj(self) -> nn.Module:
        """Forward to the parallel reward branch's gate projection."""
        return self.reward_branch.reward_gate_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        reward_values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        r"""forward(hidden_states: torch.Tensor, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=None, reward_values=None, **kwargs) -> torch.FloatTensor

        Apply the selected token-mixer block, MLP, and optional parallel
        reward branch with residual connections.

        Args:
            hidden_states (torch.Tensor): Input ``[B, T, D]``.
            position_embeddings (tuple | None): ``(cos, sin)`` RoPE cache.
            attention_mask (torch.Tensor | None): Padding mask.
            position_ids (torch.LongTensor | None): Position indices.
            past_key_values (Cache | None): KV cache for decoding.
            reward_values (torch.Tensor | None): Reward signal for the
                parallel reward branch.
            **kwargs: Extra HF kwargs.

        Returns:
            torch.FloatTensor: ``[B, T, D]`` residual output.
        """
        residual = hidden_states
        x_norm = self.input_layernorm(hidden_states)

        if hasattr(self, "linear_attn"):
            main_out = self.linear_attn(
                hidden_states=x_norm,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                reward_values=reward_values,
                **kwargs,
            )
        else:
            main_out, _ = self.self_attn(
                hidden_states=x_norm,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        mixed = main_out

        if hasattr(self, "reward_branch"):
            reward_out = self.reward_branch(
                hidden_states=x_norm,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                reward_values=reward_values,
                **kwargs,
            )
            mixed = mixed + reward_out

        if self.training and self.hidden_dropout > 0.0:
            mixed = F.dropout(mixed, p=self.hidden_dropout, training=True)

        hidden_states = residual + mixed

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.training and self.hidden_dropout > 0.0:
            hidden_states = F.dropout(
                hidden_states, p=self.hidden_dropout, training=True
            )

        hidden_states = residual + hidden_states

        result: torch.FloatTensor = hidden_states
        return result
