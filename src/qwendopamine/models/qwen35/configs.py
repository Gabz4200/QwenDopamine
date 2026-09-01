"""Qwen3.5 configuration classes."""

from __future__ import annotations

from typing import Any, ClassVar

from huggingface_hub.dataclasses import strict
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLVisionConfig,
)


@strict
class Qwen3_5TextConfig(Qwen3NextConfig):
    r"""Text configuration for Qwen3.5 models."""
    model_type = "qwen3_5_text"
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
class Qwen3_5VisionConfig(Qwen3VLVisionConfig):
    r"""Vision configuration for Qwen3.5 models."""
    model_type = "qwen3_5_vision"
    deepstack_visual_indexes = AttributeError()


@strict
class Qwen3_5Config(Qwen3VLConfig):
    """Master configuration for Qwen3.5 multimodal models."""
    model_type = "qwen3_5"
    sub_configs: ClassVar[dict[str, type]] = {
        "text_config": Qwen3_5TextConfig,
        "vision_config": Qwen3_5VisionConfig,
    }
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
