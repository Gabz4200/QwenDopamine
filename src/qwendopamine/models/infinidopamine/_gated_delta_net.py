"""InfiniDopamineGatedDeltaNet: linear-attention layer with adaptive gating.

Moved from ``decoder_layer.py`` for size. The decorator stack
``@use_kernel_forward_from_hub`` / ``@use_kernelized_func`` is preserved.
"""

from __future__ import annotations

from typing import Any, Unpack

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from transformers.cache_utils import Cache
from transformers.integrations import (
    use_kernel_forward_from_hub,
    use_kernelized_func,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextGatedDeltaNet,
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
        self._swa_mask_cache: dict[
            tuple[int, int, torch.device, torch.dtype], torch.Tensor
        ] = {}
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

    def _get_swa_mask(
        self,
        seq_len: int,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a sliding-window causal mask, caching by (seq_len, sliding_window, device, dtype)."""
        key = (seq_len, sliding_window, device, dtype)
        cached = self._swa_mask_cache.get(key)
        if cached is None or cached.shape[0] < seq_len:
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=device, dtype=dtype
            )
            mask = torch.triu(mask, diagonal=1)
            for i in range(seq_len):
                lo = max(0, i - sliding_window + 1)
                mask[i, :lo] = float("-inf")
            self._swa_mask_cache[key] = mask
            return mask
        return cached[:seq_len, :seq_len]

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
            NotImplementedError: Always — this layer uses fused
                projections that do not need reordering.
        """
        raise NotImplementedError(
            "InfiniDopamineGatedDeltaNet uses fused QKV projections; "
            "no checkpoint reordering is required."
        )

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
        result: torch.Tensor = output
        return result

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
