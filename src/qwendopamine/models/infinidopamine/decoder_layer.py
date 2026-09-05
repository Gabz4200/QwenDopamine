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

from typing import Any, ClassVar

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
from qwendopamine.models.infinidopamine._mlp import InfiniDopamineMLP
from qwendopamine.models.infinidopamine._norm import InfiniDopamineRMSNorm
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

    _LINEAR_BLOCK_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"linear_attention", "gdn2", "gdn"}
    )
    _ATTENTION_BLOCK_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"full_attention", "sliding_attention"}
    )
    _REWARD_BLOCK_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "gated_reward_net",
            "reinforced_delta",
            "reward_net",
            "reward_linear_attention",
        }
    )

    def __init__(self, config: InfiniDopamineTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_dropout = getattr(
            config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
        )
        layer_types: Any = config.layer_types
        self.block_type = layer_types[layer_idx]

        if self.block_type in self._LINEAR_BLOCK_TYPES:
            self.linear_attn = InfiniDopamineGatedDeltaNet(config, layer_idx)
        elif self.block_type in self._ATTENTION_BLOCK_TYPES:
            self.self_attn = InfiniDopamineAttention(config, layer_idx)
        elif self.block_type in self._REWARD_BLOCK_TYPES:
            self.linear_attn = InfiniDopamineGatedRewardNet(config, layer_idx)
        else:
            raise ValueError(
                f"Unsupported InfiniDopamine block_type '{self.block_type}' at "
                f"layer_idx={layer_idx}. Expected one of "
                f"{sorted(self._LINEAR_BLOCK_TYPES | self._ATTENTION_BLOCK_TYPES | self._REWARD_BLOCK_TYPES)}."
            )

        if self._has_parallel_reward(config, layer_idx):
            self._init_parallel_reward_branch(config, layer_idx)

        if (
            (getattr(config, "num_experts", None) or 0) > 0
            and layer_idx not in getattr(config, "mlp_only_layers", [])
            and (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = InfiniDopamineMLP(config, config.intermediate_size)
        self.input_layernorm = InfiniDopamineRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = InfiniDopamineRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @classmethod
    def _has_parallel_reward(
        cls, config: InfiniDopamineTextConfig, layer_idx: int
    ) -> bool:
        r"""Whether the parallel reward branch is enabled for this layer.

        Resolution order:

        1. ``config.parallel_reward_layers`` is the explicit allow-list.
        2. ``config.use_parallel_reward`` opts in to the implicit rule of
           attaching the branch to attention-only layers
           (``full_attention`` / ``sliding_attention``).
        """
        explicit_layers = tuple(getattr(config, "parallel_reward_layers", ()) or ())
        if explicit_layers:
            return layer_idx in explicit_layers
        if not getattr(config, "use_parallel_reward", False):
            return False
        layer_types: Any = config.layer_types
        return layer_types[layer_idx] in cls._ATTENTION_BLOCK_TYPES

    def _init_parallel_reward_branch(
        self, config: InfiniDopamineTextConfig, layer_idx: int
    ) -> None:
        r"""Build the parallel reward branch + data-dependent gate.

        The branch shares the same normalized input as the main mixer. The
        gate starts near zero (``sigmoid(-5) ≈ 0.0067``) so the dopamine
        contribution does not perturb a pretrained main mixer before the
        gating parameters learn a useful scale.
        """
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
            reward_out = self.reward_branch_norm(reward_out)
            gate = torch.sigmoid(self.reward_gate_proj(x_norm))
            mixed = mixed + gate * reward_out

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
