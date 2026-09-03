"""Decoder layer and attention components for InfiniDopamine models."""

from __future__ import annotations

from typing import Any, ClassVar, Unpack

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from transformers.cache_utils import Cache
from transformers.integrations import (
    use_kernel_forward_from_hub,
    use_kernelized_func,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextMLP,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from transformers.utils.generic import TransformersKwargs

from qwendopamine.models.core.normalization import apply_mask_to_padding_states
from qwendopamine.models.gdn2 import torch_chunk_gdn2, torch_recurrent_gdn2
from qwendopamine.models.infinidopamine.configs import (
    InfiniDopamineConfig,
    InfiniDopamineTextConfig,
)
from qwendopamine.models.reinforced import (
    GatedRewardNet,
    GatedRewardNetConfig,
)


@use_kernel_forward_from_hub("InfiniDopamineGatedDeltaNet")
@use_kernelized_func(
    [
        torch_chunk_gdn2,
        torch_recurrent_gdn2,
        causal_conv1d_fn,
        causal_conv1d_update,
    ]
)
class InfiniDopamineGatedDeltaNet(Qwen3NextGatedDeltaNet):
    r"""InfiniDopamineGatedDeltaNet(config, layer_idx) -> None

    InfiniDopamine linear-attention layer with adaptive gating and
    per-head gate entropy monitoring.

    Args:
        config (InfiniDopamineConfig | InfiniDopamineTextConfig): Layer config.
        layer_idx (int): Layer index for cache disambiguation.
    """

    def __init__(
        self,
        config: InfiniDopamineConfig | InfiniDopamineTextConfig,
        layer_idx: int,
    ) -> None:
        super().__init__(config, layer_idx)

        del self.in_proj_qkvz
        del self.in_proj_ba

        self.sliding_window = getattr(config, "sliding_window", 1024)
        self.attention_dropout = getattr(
            config, "attention_dropout", getattr(config, "attention_dropout_prob", 0.0)
        )
        self.hidden_dropout = getattr(
            config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
        )
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(
            self.hidden_size, self.num_v_heads * self.head_k_dim, bias=False
        )
        self.in_proj_w = nn.Linear(
            self.hidden_size, self.num_v_heads * self.head_v_dim, bias=False
        )
        self.in_proj_gate = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        nn.init.zeros_(self.in_proj_gate.weight)
        self.betas = nn.Parameter(torch.zeros(1, 1, self.num_v_heads, 1))
        self.last_gate: torch.Tensor | None = None

        self._register_load_state_dict_pre_hook(self._convert_gdn1_weights_hook)

    def _convert_gdn1_weights_hook(
        self, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        b_key = prefix + "in_proj_b.weight"
        w_key = prefix + "in_proj_w.weight"
        ba_key = prefix + "in_proj_ba.weight"
        qkvz_key = prefix + "in_proj_qkvz.weight"
        qkv_key = prefix + "in_proj_qkv.weight"
        z_key = prefix + "in_proj_z.weight"
        betas_key = prefix + "betas"
        gate_proj_key = prefix + "in_proj_gate.weight"

        if betas_key not in state_dict:
            state_dict[betas_key] = self.betas.data.clone()

        if gate_proj_key not in state_dict:
            state_dict[gate_proj_key] = self.in_proj_gate.weight.data.clone()

        if qkvz_key in state_dict:
            qkvz_weight = state_dict.pop(qkvz_key)
            conv_dim = self.key_dim * 2 + self.value_dim
            state_dict[qkv_key] = qkvz_weight[:conv_dim]
            state_dict[z_key] = qkvz_weight[conv_dim:]

        if ba_key in state_dict:
            ba_weight = state_dict.pop(ba_key)
            b_weight, a_weight = torch.chunk(ba_weight, 2, dim=0)
            state_dict[prefix + "in_proj_a.weight"] = a_weight
            state_dict[b_key] = repeat(
                b_weight, "h d -> (h k) d", k=self.head_k_dim
            ).contiguous()
            state_dict[w_key] = repeat(
                b_weight, "h d -> (h v) d", v=self.head_v_dim
            ).contiguous()
        elif b_key in state_dict and state_dict[b_key].shape[0] == self.num_v_heads:
            b_weight = state_dict[b_key]
            state_dict[b_key] = repeat(
                b_weight, "h d -> (h k) d", k=self.head_k_dim
            ).contiguous()
            if w_key not in state_dict:
                state_dict[w_key] = repeat(
                    b_weight, "h d -> (h v) d", v=self.head_v_dim
                ).contiguous()

    def fix_query_key_value_ordering(self) -> None:
        r"""fix_query_key_value_ordering() -> None

        No-op required by HF checkpoint loading.

        Raises:
            AttributeError: Always — this layer uses fused projections.
        """
        raise AttributeError("Not needed for InfiniDopamine Series")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""forward(hidden_states: torch.Tensor, cache_params: Cache | None = None, attention_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor

        Apply Gated Delta Rule 2 recurrence to hidden states.

        Args:
            hidden_states (torch.Tensor): Input ``[B, T, D]``.
            cache_params (Cache | None): HF cache for decoding state.
            attention_mask (torch.Tensor | None): Padding mask ``[B, T]``.
            **kwargs: Extra HF kwargs.

        Returns:
            torch.Tensor: ``[B, T, D]`` output.
        """
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        )

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        w = self.in_proj_w(hidden_states)
        a = self.in_proj_a(hidden_states)

        if (
            use_precomputed_states
            and seq_len == 1
            and not cache_params.layers[self.layer_idx].record_past
        ):
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = b.sigmoid().reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
        w = w.sigmoid().reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        g = g.unsqueeze(-1).expand(-1, -1, -1, self.head_k_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            groups = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(groups, dim=2)
            key = key.repeat_interleave(groups, dim=2)

        recurrent_state = (
            cache_params.layers[self.layer_idx].recurrent_states[0]
            if use_precomputed_states
            else None
        )
        if use_precomputed_states and seq_len == 1:
            gdn2_attn_out, last_recurrent_state = torch_recurrent_gdn2(
                query,
                key,
                value,
                g=g,
                b=b,
                w=w,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )
        else:
            gdn2_attn_out, last_recurrent_state = torch_chunk_gdn2(
                query,
                key,
                value,
                g=g,
                b=b,
                w=w,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                chunk_size=min(seq_len, 64),
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # Local Sliding Window Attention (SWA) stream sharing QKV projections
        q_heads = query.transpose(1, 2)
        k_heads = key.transpose(1, 2)
        v_heads = value.transpose(1, 2)

        scale = 1.0 / (self.head_k_dim**0.5)
        scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) * scale

        row_idx = torch.arange(seq_len, device=scores.device).unsqueeze(-1)
        col_idx = torch.arange(seq_len, device=scores.device).unsqueeze(0)
        dist = row_idx - col_idx
        causal_mask = dist >= 0
        if self.sliding_window is not None and self.sliding_window > 0:
            swa_mask = causal_mask & (dist < self.sliding_window)
        else:
            swa_mask = causal_mask

        min_dtype_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~swa_mask.unsqueeze(0).unsqueeze(0), min_dtype_val)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                scores = scores.masked_fill(
                    ~attention_mask.bool().unsqueeze(1).unsqueeze(2), min_dtype_val
                )
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_heads.dtype)
        if self.training and self.attention_dropout > 0.0:
            attn_weights = F.dropout(
                attn_weights, p=self.attention_dropout, training=True
            )
        swa_attn_out = torch.matmul(attn_weights, v_heads).transpose(1, 2)

        # Data-dependent Infini-attention per-head gate deciding between SWA and GDN-2
        gate_logits = self.betas + self.in_proj_gate(hidden_states).unsqueeze(-1)
        attn_gate = torch.sigmoid(gate_logits)
        if self.training:
            self.last_gate = attn_gate

        core_attn_out = attn_gate * swa_attn_out + (1.0 - attn_gate) * gdn2_attn_out
        if self.training and self.hidden_dropout > 0.0:
            core_attn_out = F.dropout(
                core_attn_out, p=self.hidden_dropout, training=True
            )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        if self.training and self.hidden_dropout > 0.0:
            output = F.dropout(output, p=self.hidden_dropout, training=True)
        return output

    def get_gate_regularization_loss(
        self, target: float = 0.5, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Compute mean squared deviation of data-dependent routing gates from target balance.

        Regularizes sigmoid(gate_logits) toward 50/50 balance early in training,
        preventing early collapse to either pure SWA or pure GDN-2 before
        the state representation stabilizes.
        """
        if hidden_states is not None:
            gate_logits = self.betas + self.in_proj_gate(hidden_states).unsqueeze(-1)
            gate = torch.sigmoid(gate_logits)
        elif self.last_gate is not None:
            gate = self.last_gate
        else:
            gate = torch.sigmoid(self.betas)
        return torch.mean((gate - target) ** 2)

    def get_gate_entropy(
        self, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Compute Shannon entropy of the routing gate distribution across heads and tokens.

        Maximum entropy ln(2) ≈ 0.693 occurs at 50/50 balance (sigmoid(gate_logits) = 0.5).
        """
        if hidden_states is not None:
            gate_logits = self.betas + self.in_proj_gate(hidden_states).unsqueeze(-1)
            gate = torch.sigmoid(gate_logits)
        elif self.last_gate is not None:
            gate = self.last_gate
        else:
            gate = torch.sigmoid(self.betas)
        # Sigmoid maps R to (0, 1) strictly, so the only failure mode is NaN
        # propagating from upstream. Replace NaN/inf in the gate value with
        # 0.5 (the maximum-entropy point) so the entropy is well-defined.
        # Clamping the gate at 1e-6 / 1-1e-6 would silently mask the NaN
        # because the gradient through clamp is 0 outside the interval,
        # hiding the real failure from tensor inspection.
        finite_gate = torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0).clamp(
            1e-6, 1.0 - 1e-6
        )
        entropy = -(
            finite_gate * torch.log(finite_gate)
            + (1.0 - finite_gate) * torch.log1p(-finite_gate)
        )
        return torch.mean(entropy)


class InfiniDopamineGatedRewardNet(GatedRewardNet):
    r"""InfiniDopamineGatedRewardNet(config, layer_idx, k_stats=6, **kwargs) -> None

    InfiniDopamine reward-augmented variant of :class:`GatedRewardNet`.

    Args:
        config (InfiniDopamineConfig | InfiniDopamineTextConfig): Layer config.
        layer_idx (int): Layer index for cache disambiguation.
        k_stats (int): Number of reward statistics. Default: ``6``.
        **kwargs: Extra args forwarded to :class:`GatedRewardNet`.
    """

    def __init__(
        self,
        config: InfiniDopamineConfig | InfiniDopamineTextConfig,
        layer_idx: int,
        k_stats: int = 6,
        **kwargs: Any,
    ) -> None:
        reward_net_config = GatedRewardNetConfig(
            hidden_size=config.hidden_size,
            k_stats=k_stats,
            layer_idx=layer_idx,
            conv_size=getattr(config, "linear_conv_kernel_dim", 4),
            norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            reward_dropout=getattr(config, "reward_dropout", 0.0),
            advantage_dropout=getattr(config, "advantage_dropout", 0.0),
            hidden_dropout=getattr(
                config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
            ),
            memory_rank=getattr(config, "reward_memory_rank", None),
            **kwargs,
        )
        super().__init__(reward_net_config)
        self.config = config
        self.key_dim = getattr(config, "linear_key_head_dim", 128) * getattr(
            config, "linear_num_key_heads", 16
        )
        self.value_dim = getattr(config, "linear_value_head_dim", 128) * getattr(
            config, "linear_num_value_heads", 32
        )
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self._register_load_state_dict_pre_hook(self._convert_gdn1_weights_hook)

    def _convert_gdn1_weights_hook(
        self, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        qkvz_key = prefix + "in_proj_qkvz.weight"
        qkv_key = prefix + "in_proj_qkv.weight"
        ba_key = prefix + "in_proj_ba.weight"
        b_key = prefix + "in_proj_b.weight"
        a_key = prefix + "in_proj_a.weight"
        z_key = prefix + "in_proj_z.weight"
        conv_key = prefix + "conv1d.weight"
        dt_key = prefix + "dt_bias"
        alog_key = prefix + "A_log"
        norm_key = prefix + "norm.weight"
        out_key = prefix + "out_proj.weight"

        is_gdn1 = any(
            k in state_dict
            for k in (qkvz_key, qkv_key, ba_key, b_key, a_key, z_key, out_key)
        )
        if not is_gdn1:
            return

        if out_key in state_dict:
            out_w = state_dict.pop(out_key)
            if out_w.shape != self.output_proj.weight.shape:
                if (
                    out_w.shape[0] == self.output_proj.weight.shape[0]
                    and out_w.shape[1] >= self.output_proj.weight.shape[1]
                ):
                    out_w = out_w[:, : self.output_proj.weight.shape[1]]
                else:
                    out_w = self.output_proj.weight.data.clone()
            state_dict[prefix + "output_proj.weight"] = out_w

        if qkvz_key in state_dict:
            qkvz = state_dict.pop(qkvz_key)
            qkv = qkvz[: self.conv_dim]
        elif qkv_key in state_dict:
            qkv = state_dict.pop(qkv_key)
        else:
            qkv = None

        if qkv is not None:
            q_w, k_w, v_w = torch.split(
                qkv, [self.key_dim, self.key_dim, self.value_dim], dim=0
            )
            if q_w.shape[0] == self.hidden_size:
                state_dict[prefix + "delta_layer.q_proj.weight"] = q_w
            if k_w.shape[0] == self.hidden_size:
                state_dict[prefix + "delta_layer.memory_core.k_proj.weight"] = k_w
            if v_w.shape[0] == self.hidden_size:
                state_dict[prefix + "delta_layer.memory_core.v_proj.weight"] = v_w

        for old_k in (
            ba_key,
            b_key,
            a_key,
            z_key,
            conv_key,
            dt_key,
            alog_key,
            norm_key,
        ):
            state_dict.pop(old_k, None)

        for param_name, param in self.named_parameters():
            full_key = prefix + param_name
            if full_key not in state_dict:
                state_dict[full_key] = param.data.clone()

    def _update_reward_cache(
        self,
        cache_params: Cache,
        new_cache: dict[str, Any],
    ) -> None:
        r"""Write the freshly computed reward state back into ``cache_params``.

        Uses reward-specific cache fields so the GDN-2 branch state is never
        clobbered. Falls back to the generic field names when the cache
        layer does not yet expose the dedicated reward fields (e.g. dict
        caches returned by a non-HF caller).
        """
        layer_idx = self.layer_idx
        if layer_idx is None:
            return
        if not hasattr(cache_params, "layers"):
            return
        layer_cache = cache_params.layers[layer_idx]
        layer_cache.reward_recurrent_state = new_cache["recurrent_state"]
        layer_cache.reward_value_baseline = new_cache["value_baseline"]
        layer_cache.reward_conv_states = new_cache["conv_state"]
        # Reward normalisation EMA statistics persist when present.
        running_mean = new_cache.get("running_mean")
        running_std = new_cache.get("running_std")
        if running_mean is not None:
            layer_cache.reward_running_mean = running_mean
        if running_std is not None:
            layer_cache.reward_running_std = running_std

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        reward_values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""forward(hidden_states, cache_params=None, attention_mask=None, reward_values=None, **kwargs) -> torch.Tensor

        Apply GatedRewardNet recurrence and persist reward cache.

        Args:
            hidden_states (torch.Tensor): Input ``[B, T, D]``.
            cache_params (Cache | None): HF cache to read/write states.
            attention_mask (torch.Tensor | None): Padding mask.
            reward_values (torch.Tensor | None): Reward input ``[B, T]``.
            **kwargs: Extra kwargs forwarded to the parent.

        Returns:
            torch.Tensor: ``[B, T, D]`` output.
        """
        use_cache = kwargs.pop("use_cache", cache_params is not None)
        out, _, new_cache = super().forward(
            hidden_states=hidden_states,
            reward_values=reward_values,
            past_key_values=cache_params,
            use_cache=use_cache,
            **kwargs,
        )
        if use_cache and cache_params is not None and new_cache is not None:
            self._update_reward_cache(cache_params, new_cache)
        return out


class InfiniDopamineAttention(Qwen3NextAttention):
    r"""InfiniDopamineAttention(config, layer_idx) -> None

    Standard full-attention layer with sliding-window support.

    Args:
        config (InfiniDopamineTextConfig): Layer configuration.
        layer_idx (int): Layer index.
    """

    def __init__(self, config: InfiniDopamineTextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.sliding_window = getattr(config, "sliding_window", 1024)
        self.attention_dropout = getattr(
            config, "attention_dropout", getattr(config, "attention_dropout_prob", 0.0)
        )


class InfiniDopamineMLP(Qwen3NextMLP):
    r"""InfiniDopamineMLP(config, intermediate_size) -> None

    MLP block with hidden-state dropout.

    Args:
        config (InfiniDopamineConfig): Layer configuration.
        intermediate_size (int): Feed-forward hidden dimension.
    """

    def __init__(self, config: InfiniDopamineConfig, intermediate_size: int) -> None:
        super().__init__(config, intermediate_size)
        self.intermediate_size = intermediate_size
        self.hidden_dropout = getattr(
            config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        r"""forward(hidden_state: torch.Tensor) -> torch.Tensor

        Apply gated MLP with optional training dropout.

        Args:
            hidden_state (torch.Tensor): Input ``[..., D]``.

        Returns:
            torch.Tensor: ``[..., D]`` output.
        """
        gate = self.act_fn(self.gate_proj(hidden_state))
        if self.training and self.hidden_dropout > 0.0:
            gate = F.dropout(gate, p=self.hidden_dropout, training=True)
        up = self.up_proj(hidden_state)
        down = self.down_proj(gate * up)
        if self.training and self.hidden_dropout > 0.0:
            down = F.dropout(down, p=self.hidden_dropout, training=True)
        return down


class InfiniDopamineRMSNorm(Qwen3NextRMSNorm):
    r"""InfiniDopamineRMSNorm: RMS normalization (inherits from
    :class:`Qwen3NextRMSNorm`).
    """


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
        self.block_type = config.layer_types[layer_idx]

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
        return config.layer_types[layer_idx] in cls._ATTENTION_BLOCK_TYPES

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
        **kwargs: Unpack[TransformersKwargs],
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

        return hidden_states
