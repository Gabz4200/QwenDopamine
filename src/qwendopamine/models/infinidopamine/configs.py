"""InfiniDopamine configuration classes."""

from __future__ import annotations

from typing import Any, ClassVar

from huggingface_hub.dataclasses import strict
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLVisionConfig,
)


@strict
class InfiniDopamineTextConfig(Qwen3NextConfig):
    r"""Text configuration for InfiniDopamine models."""
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
    use_parallel_reward: bool = False
    r"""When True, attach the GatedRewardNet parallel branch to every
    attention-only layer (``full_attention``/``sliding_attention``). When
    False, only the explicit ``parallel_reward_layers`` allow-list is used."""
    parallel_reward_layers: tuple[int, ...] = ()
    r"""Layer indices that opt-in to the parallel GatedRewardNet branch.
    When non-empty this overrides the ``use_parallel_reward`` heuristic."""
    reward_gate_init_bias: float = -5.0
    r"""Initial bias of the data-dependent reward gate ``sigmoid(W x + b)``.
    ``sigmoid(-5) ≈ 0.0067`` so the dopamine branch starts near silent and
    cannot destabilize a pretrained main mixer."""
    reward_memory_rank: int | None = None
    r"""Optional low-rank factorization of the parallel reward branch's
    ``d × d`` fast-weight state. ``None`` keeps the dense matrix. With
    ``r=64`` and ``d=4096`` the per-layer recurrent state drops from
    ``B·d·d`` to ``2·B·d·r`` parameters."""
    parallel_reward_gate_loss_weight: float = 0.0
    r"""Optional MSE penalty that keeps the parallel reward gate close to
    its initialisation (``sigmoid(reward_gate_init_bias)``). ``0`` disables
    it. With weight 1.0 the dopamine branch stays effectively silent until
    the rest of the model has stabilised."""

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
    r"""Vision configuration for InfiniDopamine models."""
    model_type = "infinidopamine_vision"
    deepstack_visual_indexes = AttributeError()


@strict
class InfiniDopamineConfig(Qwen3VLConfig):
    """Master configuration for InfiniDopamine multimodal models."""
    model_type = "infinidopamine"
    sub_configs: ClassVar[dict[str, type]] = {
        "text_config": InfiniDopamineTextConfig,
        "vision_config": InfiniDopamineVisionConfig,
    }
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
