"""Decoder layer and attention components for Qwen3.5 models."""

from __future__ import annotations

from typing import Any, Unpack

import torch
import torch.nn.functional as F
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
    causal_conv1d_fn,
    causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.utils.generic import TransformersKwargs

from qwendopamine.models.core.normalization import apply_mask_to_padding_states
from qwendopamine.models.qwen35.configs import Qwen3_5Config, Qwen3_5TextConfig


@use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")
@use_kernelized_func(
    [
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
        causal_conv1d_fn,
        causal_conv1d_update,
    ]
)
class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
    r"""Qwen3_5GatedDeltaNet(config: Qwen3_5Config | Qwen3_5TextConfig, layer_idx: int) -> None

    Qwen3.5 linear-attention layer backed by Gated Delta Rule 2.

    Replaces the upstream ``in_proj_qkvz`` / ``in_proj_ba`` with a fused
    ``in_proj_qkv`` projection.

    Args:
        config (Qwen3_5Config | Qwen3_5TextConfig): Qwen3.5 configuration.
        layer_idx (int): Layer index for cache disambiguation.

    Dimension convention (paper + upstream Qwen3Next):

      - ``hidden_size``         : model width ``D``
      - ``num_v_heads``         : number of value heads ``H_v``
      - ``num_k_heads``         : number of key heads ``H_k``
      - ``head_k_dim``          : per-key-head dim ``d_k``
      - ``head_v_dim``          : per-value-head dim ``d_v``
      - ``key_dim = H_k * d_k``  (concatenated keys)
      - ``value_dim = H_v * d_v``  (concatenated values)

    The upstream ``Qwen3NextGatedDeltaNet`` flattens ``b`` and ``g`` to
    one gate per head (``H_v``). The Qwen3.5 fork inherits that
    parameterisation; it is **not** the per-channel paper convention
    used by the GDN-2 reference and the GatedDeltaNet2 module.
    """

    def __init__(
        self, config: Qwen3_5Config | Qwen3_5TextConfig, layer_idx: int
    ) -> None:
        super().__init__(config, layer_idx)

        del self.in_proj_qkvz
        del self.in_proj_ba

        # QKV_SPLIT tells the forward how to slice the fused QKV
        # projection. Keep the order and sizes in lockstep with
        # ``self.in_proj_qkv``'s ``out_features``; mismatches here
        # produce silently-garbage output rather than a clean error.
        self.QKV_SPLIT: tuple[int, int, int] = (
            self.key_dim,
            self.key_dim,
            self.value_dim,
        )
        self.in_proj_qkv = nn.Linear(
            self.hidden_size,
            self.QKV_SPLIT[0] + self.QKV_SPLIT[1] + self.QKV_SPLIT[2],
            bias=False,
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def fix_query_key_value_ordering(self) -> None:
        r"""fix_query_key_value_ordering() -> None

        No-op required by HF checkpoint loading.

        Raises:
            NotImplementedError: Always — this layer uses fused
                projections that do not need reordering.
        """
        raise NotImplementedError(
            "Qwen3.5GatedDeltaNet uses fused QKV projections; no "
            "checkpoint reordering is required."
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""forward(hidden_states: torch.Tensor, cache_params: Cache | None = None, attention_mask: torch.Tensor | None = None, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor

        Apply Gated Delta Rule 2 recurrence to hidden states.

        Args:
            hidden_states (torch.Tensor): Input ``[B, T, D]``.
            cache_params (Cache | None): HF cache for decoding state.
            attention_mask (torch.Tensor | None): Padding mask ``[B, T]``.
            **kwargs: Extra HF kwargs (``cu_seq_lens_q``, etc.).

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
        a = self.in_proj_a(hidden_states)

        if (
            use_precomputed_states
            and seq_len == 1
            and not cache_params.layers[self.layer_idx].record_past
        ):
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
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

            # Drop the additional previous states
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            list(self.QKV_SPLIT),
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = (
            cache_params.layers[self.layer_idx].recurrent_states[0]
            if use_precomputed_states
            else None
        )
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        result: torch.Tensor = output
        return result


class Qwen3_5Attention(Qwen3NextAttention):
    r"""Qwen3_5Attention: standard full-attention layer (inherits from
    :class:`Qwen3NextAttention`.
    """


class Qwen3_5MLP(Qwen3NextMLP):
    r"""Qwen3_5MLP(config: Qwen3_5Config, intermediate_size: int) -> None

    MLP block wrapping the upstream :class:`Qwen3NextMLP` with an explicit
    intermediate size.

    Args:
        config (Qwen3_5Config): Qwen3.5 configuration.
        intermediate_size (int): Feed-forward hidden dimension.
    """

    def __init__(self, config: Qwen3_5Config, intermediate_size: int) -> None:
        super().__init__(config, intermediate_size)
        self.intermediate_size = intermediate_size


class Qwen3_5RMSNorm(Qwen3NextRMSNorm):
    r"""Qwen3_5RMSNorm: RMS normalization (inherits from
    :class:`Qwen3NextRMSNorm`.
    """


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    r"""Qwen3_5DecoderLayer(config: Qwen3_5TextConfig, layer_idx: int) -> None

    Decoder layer selecting GDN-2 or standard attention per ``layer_types``.

    Args:
        config (Qwen3_5TextConfig): Qwen3.5 text configuration.
        layer_idx (int): Layer index.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_types: Any = config.layer_types
        self.block_type = layer_types[layer_idx]
        # Qwen3Next supports additional block types (``sliding_attention``
        # and any future ones). We only specialise the two we know how
        # to run; anything else must be handled upstream or this
        # layer construction is invalid.
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise NotImplementedError(
                f"Qwen3.5DecoderLayer does not support block_type={self.block_type!r}. "
                "Supported: 'linear_attention', 'full_attention'."
            )
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        r"""forward(hidden_states: torch.Tensor, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=None, **kwargs) -> torch.FloatTensor

        Apply the selected token-mixer block and MLP residual sublayers.

        Args:
            hidden_states (torch.Tensor): Input ``[B, T, D]``.
            position_embeddings (tuple | None): ``(cos, sin)`` RoPE cache.
            attention_mask (torch.Tensor | None): Padding mask.
            position_ids (torch.LongTensor | None): Position indices.
            past_key_values (Cache | None): KV cache for decoding.
            **kwargs: Extra HF kwargs.

        Returns:
            torch.FloatTensor: ``[B, T, D]`` residual output.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif self.block_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            # The constructor raises NotImplementedError for unknown
            # block types, so reaching this branch means a subclass
            # mutated ``self.block_type`` post-init. Fail loudly.
            raise NotImplementedError(
                f"Unexpected block_type={self.block_type!r} at forward time."
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        result: torch.FloatTensor = hidden_states
        return result
