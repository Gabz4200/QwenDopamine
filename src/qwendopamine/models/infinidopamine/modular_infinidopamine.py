# Copyright 2025 The Qwen Team, InfiniDopamine Authors, and The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch InfiniDopamine model."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from qwendopamine.models.gdn2 import torch_chunk_gdn2, torch_recurrent_gdn2
from qwendopamine.models.gdn2.reinforced_delta import GatedRewardNet

try:
    from huggingface_hub.dataclasses import strict as _hf_strict

    def strict(cls: Any) -> Any:
        try:
            return _hf_strict(cls)
        except Exception:  # noqa: BLE001
            return cls
except ImportError:

    def strict(cls: Any) -> Any:
        return cls


try:
    from transformers import initialization as init
except ImportError:

    class _InitFallback:
        @staticmethod
        def ones_(tensor: torch.Tensor) -> torch.Tensor:
            return nn.init.ones_(tensor)

        @staticmethod
        def copy_(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            return target.copy_(source)

    init = _InitFallback()  # type: ignore[assignment]

try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:

    class Cache:  # type: ignore[no-redef]
        pass

    class DynamicCache(Cache):  # type: ignore[no-redef]
        pass


try:
    from transformers.integrations import (
        use_kernel_forward_from_hub as _hf_use_kernel_forward_from_hub,
    )
    from transformers.integrations import (
        use_kernelized_func as _hf_use_kernelized_func,
    )

    def use_kernel_forward_from_hub(*args: Any, **kwargs: Any) -> Any:
        try:
            inner = _hf_use_kernel_forward_from_hub(*args, **kwargs)
        except Exception:  # noqa: BLE001

            def noop_decorator(fn: Any) -> Any:
                return fn

            return noop_decorator

        def decorator(fn: Any) -> Any:
            try:
                return inner(fn)
            except Exception:  # noqa: BLE001
                return fn

        return decorator

    def use_kernelized_func(*args: Any, **kwargs: Any) -> Any:
        try:
            inner = _hf_use_kernelized_func(*args, **kwargs)
        except Exception:  # noqa: BLE001

            def noop_decorator(fn: Any) -> Any:
                return fn

            return noop_decorator

        def decorator(fn: Any) -> Any:
            try:
                return inner(fn)
            except Exception:  # noqa: BLE001
                return fn

        return decorator
except ImportError:

    def use_kernel_forward_from_hub(*args: Any, **kwargs: Any) -> Any:
        def decorator(fn: Any) -> Any:
            return fn

        return decorator

    def use_kernelized_func(*args: Any, **kwargs: Any) -> Any:
        def decorator(fn: Any) -> Any:
            return fn

        return decorator


try:
    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    try:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
    except ImportError:
        try:
            from transformers.masking_utils import create_causal_mask
        except ImportError:

            def create_causal_mask(*args: Any, **kwargs: Any) -> Any:
                return None

        def create_sliding_window_causal_mask(*args: Any, **kwargs: Any) -> Any:
            return None

    def create_recurrent_attention_mask(*args: Any, **kwargs: Any) -> Any:
        return None


try:
    from transformers.modeling_layers import (
        GenericForSequenceClassification,
        GenericForTokenClassification,
        GradientCheckpointingLayer,
    )
except ImportError:

    class GenericForSequenceClassification:  # type: ignore[no-redef]
        pass

    class GenericForTokenClassification:  # type: ignore[no-redef]
        pass

    class GradientCheckpointingLayer(nn.Module):  # type: ignore[no-redef]
        pass


try:
    from transformers.modeling_outputs import (
        BaseModelOutputWithPast,
        BaseModelOutputWithPooling,
        CausalLMOutputWithPast,
        SequenceClassifierOutputWithPast,
    )
except ImportError:

    class BaseModelOutputWithPast:  # type: ignore[no-redef]
        pass

    class BaseModelOutputWithPooling:  # type: ignore[no-redef]
        pass

    class CausalLMOutputWithPast:  # type: ignore[no-redef]
        pass

    class SequenceClassifierOutputWithPast:  # type: ignore[no-redef]
        pass


try:
    from transformers.modeling_utils import PreTrainedModel
except ImportError:

    class PreTrainedModel(nn.Module):  # type: ignore[no-redef]
        pass


try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
except Exception:  # noqa: BLE001

    class Qwen3ForCausalLM(nn.Module):  # type: ignore[no-redef]
        pass


try:
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
except Exception:  # noqa: BLE001

    class Qwen3NextConfig:  # type: ignore[no-redef]
        pass


try:
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextAttention,
        Qwen3NextGatedDeltaNet,
        Qwen3NextMLP,
        Qwen3NextModel,
        Qwen3NextPreTrainedModel,
        Qwen3NextRMSNorm,
        Qwen3NextSparseMoeBlock,
        apply_mask_to_padding_states,
        causal_conv1d_fn,
        causal_conv1d_update,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )
except Exception:  # noqa: BLE001

    class Qwen3NextAttention(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextGatedDeltaNet(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextMLP(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextSparseMoeBlock(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextModel(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextPreTrainedModel(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextRMSNorm(nn.Module):  # type: ignore[no-redef]
        pass

    apply_mask_to_padding_states = None  # type: ignore[misc, assignment]
    causal_conv1d_fn = None  # type: ignore[misc, assignment]
    causal_conv1d_update = None  # type: ignore[misc, assignment]
    torch_chunk_gated_delta_rule = None  # type: ignore[misc, assignment]
    torch_recurrent_gated_delta_rule = None  # type: ignore[misc, assignment]

if not torch.cuda.is_available():
    if torch_chunk_gated_delta_rule is not None:
        while hasattr(torch_chunk_gated_delta_rule, "__wrapped__"):
            torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule.__wrapped__
    if torch_recurrent_gated_delta_rule is not None:
        while hasattr(torch_recurrent_gated_delta_rule, "__wrapped__"):
            torch_recurrent_gated_delta_rule = (
                torch_recurrent_gated_delta_rule.__wrapped__
            )
    if causal_conv1d_fn is not None:
        while hasattr(causal_conv1d_fn, "__wrapped__"):
            causal_conv1d_fn = causal_conv1d_fn.__wrapped__
    if causal_conv1d_update is not None:
        while hasattr(causal_conv1d_update, "__wrapped__"):
            causal_conv1d_update = causal_conv1d_update.__wrapped__
    try:
        import transformers.models.qwen3_next.modeling_qwen3_next as _q3n

        for _name in [
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
            "causal_conv1d_update",
        ]:
            if hasattr(_q3n, _name):
                _fn = getattr(_q3n, _name)
                while hasattr(_fn, "__wrapped__"):
                    _fn = _fn.__wrapped__
                setattr(_q3n, _name, _fn)
    except (ImportError, AttributeError):
        pass

try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLVisionConfig,
    )
except Exception:  # noqa: BLE001
    class Qwen3VLConfig:  # type: ignore[no-redef]
        pass

    class Qwen3VLVisionConfig:  # type: ignore[no-redef]
        pass


try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
        Qwen3VLModelOutputWithPast,
        Qwen3VLTextRotaryEmbedding,
        Qwen3VLVisionModel,
        Qwen3VLVisionRotaryEmbedding,
    )
except Exception:  # noqa: BLE001
    class Qwen3VLForConditionalGeneration(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3VLModel(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3VLModelOutputWithPast:  # type: ignore[no-redef]
        pass

    class Qwen3VLTextRotaryEmbedding(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3VLVisionModel(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3VLVisionRotaryEmbedding(nn.Module):  # type: ignore[no-redef]
        pass


try:
    from typing import Unpack
except ImportError:
    try:
        from transformers.processing_utils import Unpack
    except ImportError:
        Unpack = Any  # type: ignore[misc, assignment]

try:
    from transformers.utils import (
        TransformersKwargs,
        can_return_tuple,
        logging,
    )
except ImportError:

    def can_return_tuple(fn: Any) -> Any:
        return fn

    TransformersKwargs = Any  # type: ignore[misc, assignment]
    import logging

try:
    from transformers.utils.generic import (
        accepts_precomputed_kwargs,
        merge_with_config_defaults,
    )
except ImportError:

    def accepts_precomputed_kwargs(*args: Any, **kwargs: Any) -> Any:
        def decorator(fn: Any) -> Any:
            return fn

        return decorator

    def merge_with_config_defaults(fn: Any) -> Any:
        return fn


try:
    from transformers.utils.output_capturing import capture_outputs
except ImportError:

    def capture_outputs(fn: Any) -> Any:
        return fn


try:
    from transformers.vision_utils import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )
except ImportError:
    get_vision_attention_seqlens = None  # type: ignore[misc, assignment]
    get_vision_interpolation_indices_and_weights = None  # type: ignore[misc, assignment]
    get_vision_position_ids = None  # type: ignore[misc, assignment]

logger = logging.get_logger(__name__)


@strict
class InfiniDopamineTextConfig(Qwen3NextConfig):
    r"""
    sliding_window (`int`, *optional*, defaults to 1024):
        Sliding window attention window size.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the convolution used in linear attention layers.
    linear_key_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each key head in linear attention.
    linear_value_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each value head in linear attention.
    linear_num_key_heads (`int`, *optional*, defaults to 16):
        Number of key heads used in linear attention layers.
    linear_num_value_heads (`int`, *optional*, defaults to 32):
        Number of value heads used in linear attention layers.
    """

    model_type = "infinidopamine_text"
    base_config_key = "text_config"

    base_model_tp_plan: ClassVar[dict[str, str]] = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_w": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_a": "colwise_gather_output",
        "layers.*.linear_attn.out_proj": "colwise_gather_output",
    }
    base_model_ep_plan = None  # no Moe
    ignore_keys_at_rope_validation: ClassVar[set[str]] = {
        "mrope_section",
        "mrope_interleaved",
    }

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_key_value_heads: int = 4
    sliding_window: int | None = 1024
    attention_dropout: float = 0.05
    hidden_dropout: float = 0.05
    reward_dropout: float = 0.10
    advantage_dropout: float = 0.05
    gate_loss_weight: float = 0.01
    gate_target_balance: float = 0.5

    @property
    def gate_reg_coef(self) -> float:
        return self.gate_loss_weight

    @gate_reg_coef.setter
    def gate_reg_coef(self, val: float) -> None:
        self.gate_loss_weight = val

    @property
    def attention_dropout_prob(self) -> float:
        return self.attention_dropout

    @attention_dropout_prob.setter
    def attention_dropout_prob(self, val: float) -> None:
        self.attention_dropout = val

    @property
    def hidden_dropout_prob(self) -> float:
        return self.hidden_dropout

    @hidden_dropout_prob.setter
    def hidden_dropout_prob(self, val: float) -> None:
        self.hidden_dropout = val

    @property
    def mlp_only_layers(self) -> list[int]:
        return []

    @mlp_only_layers.setter
    def mlp_only_layers(self, val: object) -> None:
        pass

    @property
    def decoder_sparse_step(self) -> int:
        return 1

    @decoder_sparse_step.setter
    def decoder_sparse_step(self, val: object) -> None:
        pass

    norm_topk_prob = AttributeError()
    moe_intermediate_size = AttributeError()
    shared_expert_intermediate_size = AttributeError()
    num_experts_per_tok = AttributeError()
    num_experts = AttributeError()
    output_router_logits = AttributeError()
    router_aux_loss_coef = AttributeError()

    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)
        if "mlp_only_layers" in self.__dict__:
            del self.__dict__["mlp_only_layers"]


@strict
class InfiniDopamineVisionConfig(Qwen3VLVisionConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 3584):
        The output hidden size of the vision model.
    num_position_embeddings (`int`, *optional*, defaults to 2304):
        The maximum sequence length that this model might ever be used with
    """

    model_type = "infinidopamine_vision"
    deepstack_visual_indexes = AttributeError()


@strict
class InfiniDopamineConfig(Qwen3VLConfig):
    model_type = "infinidopamine"
    sub_configs: ClassVar[dict[str, type]] = {
        "text_config": InfiniDopamineTextConfig,
        "vision_config": InfiniDopamineVisionConfig,
    }
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054


class InfiniDopamineVisionRotaryEmbedding(Qwen3VLVisionRotaryEmbedding):
    pass


class InfiniDopamineTextRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    def __init__(self, config: InfiniDopamineTextConfig, device: Any = None) -> None:
        super().__init__(config)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: InfiniDopamineTextConfig, device: Any = None, **kwargs: Any
    ) -> tuple[torch.Tensor, float]:
        _ = kwargs
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor


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


class InfiniDopaminePreTrainedModel(Qwen3NextPreTrainedModel):
    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
    _no_split_modules: ClassVar[list[str]] = [
        "InfiniDopamineDecoderLayer",
        "InfiniDopamineVisionBlock",
    ]
    _can_record_outputs: ClassVar[dict[str, type]] = {
        "hidden_states": InfiniDopamineDecoderLayer,
        "attentions": InfiniDopamineAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, InfiniDopamineGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device)
                .uniform_(0.01, 16)
                .log_(),
            )
            init.zeros_(module.betas)
            init.zeros_(module.in_proj_gate.weight)
        elif isinstance(module, InfiniDopamineRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, InfiniDopamineVisionRotaryEmbedding):
            inv_freq = 1.0 / (
                module.theta
                ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim)
            )
            init.copy_(module.inv_freq, inv_freq)


class InfiniDopamineVisionModel(Qwen3VLVisionModel):
    config_class = InfiniDopamineVisionConfig
    config: InfiniDopamineVisionConfig
    _no_split_modules: ClassVar[list[str]] = ["InfiniDopamineVisionBlock"]

    def __init__(self, config: Any, *inputs: Any, **kwargs: Any) -> None:
        super().__init__(config, *inputs, **kwargs)
        del self.deepstack_visual_indexes
        del self.deepstack_merger_list

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.config.spatial_merge_size,
            kwargs=kwargs,
        )
        position_ids = get_vision_position_ids(
            grid_thw, self.spatial_merge_size, kwargs=kwargs
        )
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(
            grid_thw, self.config, kwargs=kwargs
        )
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(
            1
        )
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class InfiniDopamineModelOutputWithPast(Qwen3VLModelOutputWithPast):
    pass


class InfiniDopamineTextModel(Qwen3NextModel):
    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig

    def __init__(self, config: InfiniDopamineTextConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                InfiniDopamineDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = InfiniDopamineTextRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            if past_key_values is not None:
                try:
                    past_seen_tokens = past_key_values.get_seq_length()
                except ValueError:
                    past_seen_tokens = 0
            else:
                past_seen_tokens = 0
            position_ids = (
                torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                )
                + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(
                4, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[:, None, :].expand(
                4, position_ids.shape[0], -1
            )

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            attn_mask = (
                create_sliding_window_causal_mask(**mask_kwargs)
                if getattr(self.config, "sliding_window", None) is not None
                else create_causal_mask(**mask_kwargs)
            )
            causal_mask_mapping = {
                "full_attention": attn_mask,
                "sliding_attention": attn_mask,
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        reward_values = kwargs.pop("reward_values", None)
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                reward_values=reward_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return InfiniDopamineModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""Compute total gate balance regularization loss across all GDN-2 mixer layers."""
        losses: list[torch.Tensor] = []
        for layer in self.layers[: self.config.num_hidden_layers]:
            if hasattr(layer, "linear_attn") and hasattr(
                layer.linear_attn, "get_gate_regularization_loss"
            ):
                losses.append(
                    layer.linear_attn.get_gate_regularization_loss(target=target)
                )
        if not losses:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def load_qwen35_weights(
        self,
        weights: dict[str, torch.Tensor] | nn.Module,
        strict: bool = True,
    ) -> Any:
        r"""Load pretrained Qwen3.5 (GDN-1) weights into InfiniDopamine (GDN-2 with SWA)."""
        if isinstance(weights, nn.Module):
            state_dict = weights.state_dict()
        else:
            state_dict = dict(weights)

        has_full_prefix = any(
            k.startswith(("model.language_model.", "model.", "language_model."))
            for k in state_dict
        )
        if has_full_prefix:
            remapped_state_dict: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith("model.language_model."):
                    new_k = new_k[len("model.language_model.") :]
                elif new_k.startswith("language_model."):
                    new_k = new_k[len("language_model.") :]
                elif new_k.startswith("model."):
                    new_k = new_k[len("model.") :]
                if not (
                    k.startswith(("model.visual.", "visual.", "mtp."))
                    or k == "lm_head.weight"
                ):
                    remapped_state_dict[new_k] = v
            state_dict = remapped_state_dict

        return self.load_state_dict(state_dict, strict=strict)


class InfiniDopamineModel(Qwen3VLModel):
    config_class = InfiniDopamineConfig
    _no_split_modules: ClassVar[list[str]] = [
        "InfiniDopamineDecoderLayer",
        "InfiniDopamineVisionBlock",
    ]

    def __init__(self, config: InfiniDopamineConfig) -> None:
        InfiniDopaminePreTrainedModel.__init__(self, config)
        self.visual = InfiniDopamineVisionModel(config.vision_config)
        self.language_model = InfiniDopamineTextModel(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_video_features(
        self, **super_kwargs: Any
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        return super().get_video_features(**super_kwargs)

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output: BaseModelOutputWithPooling = self.visual(
            pixel_values, image_grid_thw=image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Any, ...] | InfiniDopamineModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw=image_grid_thw, return_dict=True, **kwargs
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos,
                video_grid_thw=video_grid_thw,
                return_dict=True,
                **kwargs,
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return InfiniDopamineModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class InfiniDopamineForCausalLM(Qwen3ForCausalLM):
    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
        r"^model.visual.*",
    ]

    def __init__(self, config: InfiniDopamineTextConfig) -> None:
        super().__init__(config)
        self.model = InfiniDopamineTextModel(config)

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""Compute total gate balance regularization loss across all GDN-2 mixer layers."""
        return self.model.get_gate_regularization_loss(target=target)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        reward_values: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: CausalLMOutputWithPast = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            reward_values=reward_values,
            **kwargs,
        )
        if (
            labels is not None
            and getattr(outputs, "loss", None) is not None
            and self.training
            and getattr(self.config, "gate_loss_weight", 0.0) > 0.0
        ):
            target = getattr(self.config, "gate_target_balance", 0.5)
            gate_loss = self.get_gate_regularization_loss(target=target)
            outputs.loss = outputs.loss + self.config.gate_loss_weight * gate_loss
        return outputs

    def load_qwen35_weights(
        self,
        weights: dict[str, torch.Tensor] | nn.Module,
        strict: bool = True,
    ) -> Any:
        r"""Load pretrained Qwen3.5 (GDN-1) weights into InfiniDopamine (GDN-2 with SWA)."""
        if isinstance(weights, nn.Module):
            state_dict = weights.state_dict()
        else:
            state_dict = dict(weights)

        has_language_model_prefix = any(
            k.startswith(("model.language_model.", "language_model."))
            for k in state_dict
        )
        if has_language_model_prefix:
            remapped_state_dict: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.startswith("model.language_model."):
                    remapped_state_dict[k.replace("model.language_model.", "model.")] = v
                elif k.startswith("language_model."):
                    remapped_state_dict[k.replace("language_model.", "model.")] = v
                elif k == "lm_head.weight" or not k.startswith(("model.visual.", "visual.", "mtp.")):
                    remapped_state_dict[k] = v
            state_dict = remapped_state_dict

        return self.load_state_dict(state_dict, strict=strict)


class InfiniDopamineForTokenClassification(
    GenericForTokenClassification, InfiniDopaminePreTrainedModel
):
    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig


class InfiniDopamineForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
    ]

    def __init__(self, config: InfiniDopamineConfig) -> None:
        InfiniDopaminePreTrainedModel.__init__(self, config)
        self.model = InfiniDopamineModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    def get_video_features(
        self, **super_kwargs: Any
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        return super().get_video_features(**super_kwargs)

    def get_image_features(
        self, **super_kwargs: Any
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        return super().get_image_features(**super_kwargs)

    def load_qwen35_weights(
        self,
        weights: dict[str, torch.Tensor] | nn.Module,
        strict: bool = True,
    ) -> Any:
        r"""Load pretrained Qwen3.5 (GDN-1) weights into InfiniDopamine (GDN-2 with SWA)."""
        if isinstance(weights, nn.Module):
            state_dict = weights.state_dict()
        else:
            state_dict = dict(weights)
        return self.load_state_dict(state_dict, strict=strict)


class InfiniDopamineTextForSequenceClassification(
    GenericForSequenceClassification, InfiniDopaminePreTrainedModel
):
    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig
    input_modalities = ("text",)


class InfiniDopamineForSequenceClassification(
    GenericForSequenceClassification, InfiniDopaminePreTrainedModel
):
    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )


__all__ = [
    "InfiniDopamineConfig",
    "InfiniDopamineDecoderLayer",
    "InfiniDopamineForCausalLM",
    "InfiniDopamineForConditionalGeneration",
    "InfiniDopamineForSequenceClassification",
    "InfiniDopamineForTokenClassification",
    "InfiniDopamineGatedDeltaNet",
    "InfiniDopamineModel",
    "InfiniDopaminePreTrainedModel",
    "InfiniDopamineTextConfig",
    "InfiniDopamineTextForSequenceClassification",
    "InfiniDopamineTextModel",
    "InfiniDopamineTextRotaryEmbedding",
    "InfiniDopamineVisionConfig",
    "InfiniDopamineVisionModel",
    "InfiniDopamineVisionRotaryEmbedding",
]
