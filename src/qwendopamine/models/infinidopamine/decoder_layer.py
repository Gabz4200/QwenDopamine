"""Decoder layer and attention components for InfiniDopamine models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from qwendopamine.models._hf_compat import (
    Cache,
    GradientCheckpointingLayer,
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextMLP,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
    TransformersKwargs,
    Unpack,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
    use_kernel_forward_from_hub,
    use_kernelized_func,
)
from qwendopamine.models.gdn2 import torch_chunk_gdn2, torch_recurrent_gdn2
from qwendopamine.models.gdn2.reinforced_delta import GatedRewardNet
from qwendopamine.models.infinidopamine.configs import (
    InfiniDopamineConfig,
    InfiniDopamineTextConfig,
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
        raise AttributeError("Not needed for InfiniDopamine Series")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
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
        scores = scores.masked_fill(
            ~swa_mask.unsqueeze(0).unsqueeze(0), min_dtype_val
        )
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                scores = scores.masked_fill(
                    ~attention_mask.bool().unsqueeze(1).unsqueeze(2), min_dtype_val
                )
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(
            q_heads.dtype
        )
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

        core_attn_out = (
            attn_gate * swa_attn_out + (1.0 - attn_gate) * gdn2_attn_out
        )
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
        gate = gate.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(gate * torch.log(gate) + (1.0 - gate) * torch.log(1.0 - gate))
        return torch.mean(entropy)


class InfiniDopamineGatedRewardNet(GatedRewardNet):
    def __init__(
        self,
        config: InfiniDopamineConfig | InfiniDopamineTextConfig,
        layer_idx: int,
        k_stats: int = 6,
        **kwargs: Any,
    ) -> None:
        reward_dropout = getattr(config, "reward_dropout", 0.0)
        advantage_dropout = getattr(config, "advantage_dropout", 0.0)
        hidden_dropout = getattr(
            config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
        )
        super().__init__(
            hidden_size=config.hidden_size,
            k_stats=k_stats,
            layer_idx=layer_idx,
            conv_size=getattr(config, "linear_conv_kernel_dim", 4),
            norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            reward_dropout=reward_dropout,
            advantage_dropout=advantage_dropout,
            hidden_dropout=hidden_dropout,
            **kwargs,
        )
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        reward_values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        use_cache = kwargs.pop("use_cache", cache_params is not None)
        out, _, _ = super().forward(
            hidden_states=hidden_states,
            reward_values=reward_values,
            past_key_values=cache_params,
            use_cache=use_cache,
            **kwargs,
        )
        return out


class InfiniDopamineAttention(Qwen3NextAttention):
    def __init__(self, config: InfiniDopamineTextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.sliding_window = getattr(config, "sliding_window", 1024)
        self.attention_dropout = getattr(
            config, "attention_dropout", getattr(config, "attention_dropout_prob", 0.0)
        )


class InfiniDopamineMLP(Qwen3NextMLP):
    def __init__(self, config: InfiniDopamineConfig, intermediate_size: int) -> None:
        super().__init__(config, intermediate_size)
        self.intermediate_size = intermediate_size
        self.hidden_dropout = getattr(
            config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(hidden_state))
        if self.training and self.hidden_dropout > 0.0:
            gate = F.dropout(gate, p=self.hidden_dropout, training=True)
        up = self.up_proj(hidden_state)
        down = self.down_proj(gate * up)
        if self.training and self.hidden_dropout > 0.0:
            down = F.dropout(down, p=self.hidden_dropout, training=True)
        return down


class InfiniDopamineRMSNorm(Qwen3NextRMSNorm):
    pass


class InfiniDopamineDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: InfiniDopamineTextConfig, layer_idx: int
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_dropout = getattr(
            config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
        )
        self.block_type = config.layer_types[layer_idx]
        is_pre_attention = (
            layer_idx + 1 < len(config.layer_types)
            and config.layer_types[layer_idx + 1]
            in ("full_attention", "sliding_attention")
        )
        if self.block_type in ("linear_attention", "gdn2", "gdn"):
            if is_pre_attention:
                self.linear_attn = InfiniDopamineGatedRewardNet(
                    config, layer_idx
                )
            else:
                self.linear_attn = InfiniDopamineGatedDeltaNet(
                    config, layer_idx
                )
        elif self.block_type in (
            "gated_reward_net",
            "reinforced_delta",
            "reward_net",
            "reward_linear_attention",
        ):
            self.linear_attn = InfiniDopamineGatedRewardNet(config, layer_idx)
        elif self.block_type in ("full_attention", "sliding_attention"):
            self.self_attn = InfiniDopamineAttention(config, layer_idx)
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
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if hasattr(self, "linear_attn"):
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                reward_values=reward_values,
                **kwargs,
            )
        elif hasattr(self, "self_attn"):
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if self.training and self.hidden_dropout > 0.0:
            hidden_states = F.dropout(
                hidden_states, p=self.hidden_dropout, training=True
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.training and self.hidden_dropout > 0.0:
            hidden_states = F.dropout(
                hidden_states, p=self.hidden_dropout, training=True
            )

        hidden_states = residual + hidden_states

        return hidden_states
