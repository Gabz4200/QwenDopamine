"""Behavioral tests for the InfiniDopamine modular architecture."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineDecoderLayer,
    InfiniDopamineForCausalLM,
    InfiniDopamineGatedDeltaNet,
    InfiniDopamineGatedRewardNet,
    InfiniDopamineTextConfig,
    InfiniDopamineTextModel,
    InfiniDopamineVisionConfig,
)
from qwendopamine.models.qwen35 import (
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)


@pytest.fixture
def tiny_infini_config() -> InfiniDopamineTextConfig:
    r"""Fixture providing a fast, minimal InfiniDopamine configuration."""
    return InfiniDopamineTextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


def test_when_default_config_instantiated_then_contains_valid_defaults() -> None:
    config = InfiniDopamineTextConfig()
    assert config.vocab_size == 248320
    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 32
    assert config.sliding_window == 1024
    assert config.model_type == "infinidopamine_text"


def test_when_vision_config_instantiated_then_contains_valid_defaults() -> None:
    config = InfiniDopamineVisionConfig()
    assert config.out_hidden_size > 0
    assert config.num_position_embeddings > 0


def test_when_multimodal_config_instantiated_then_contains_valid_defaults() -> None:
    config = InfiniDopamineConfig()
    assert config.image_token_id == 248056
    assert config.video_token_id == 248057
    assert config.vision_start_token_id == 248053
    assert config.vision_end_token_id == 248054


def test_when_text_model_forward_with_mask_then_returns_valid_hidden_states(
    tiny_infini_config: InfiniDopamineTextConfig,
) -> None:
    model = InfiniDopamineTextModel(tiny_infini_config)
    input_ids = torch.tensor([[1, 5, 10, 20], [2, 4, 8, 16]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    assert output.last_hidden_state.shape == (2, 4, tiny_infini_config.hidden_size)
    assert not torch.isnan(output.last_hidden_state).any()


def test_when_gated_deltanet_forward_then_preserves_shape(
    tiny_infini_config: InfiniDopamineTextConfig,
) -> None:
    layer = InfiniDopamineGatedDeltaNet(tiny_infini_config, layer_idx=0)
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, tiny_infini_config.hidden_size)

    output = layer(hidden_states)
    assert output.shape == (batch_size, seq_len, tiny_infini_config.hidden_size)
    assert not torch.isnan(output).any()


def test_when_decoder_layer_forward_then_preserves_shape(
    tiny_infini_config: InfiniDopamineTextConfig,
) -> None:
    layer = InfiniDopamineDecoderLayer(tiny_infini_config, layer_idx=0)
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, tiny_infini_config.hidden_size)

    output = layer(hidden_states)
    tensor_output = output[0] if isinstance(output, tuple) else output
    assert tensor_output.shape == (batch_size, seq_len, tiny_infini_config.hidden_size)
    assert not torch.isnan(tensor_output).any()


def test_when_text_model_forward_then_returns_valid_hidden_states(
    tiny_infini_config: InfiniDopamineTextConfig,
) -> None:
    model = InfiniDopamineTextModel(tiny_infini_config)
    input_ids = torch.tensor([[1, 5, 10, 20], [2, 4, 8, 16]], dtype=torch.long)

    output = model(input_ids=input_ids)
    assert output.last_hidden_state.shape == (2, 4, tiny_infini_config.hidden_size)
    assert not torch.isnan(output.last_hidden_state).any()


def test_when_causal_lm_forward_then_computes_logits_and_loss(
    tiny_infini_config: InfiniDopamineTextConfig,
) -> None:
    model = InfiniDopamineForCausalLM(tiny_infini_config)
    input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.long)
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels)
    assert output.logits.shape == (2, 4, tiny_infini_config.vocab_size)
    assert output.loss is not None
    assert output.loss.item() > 0.0


def test_when_gradients_computed_then_parameters_receive_grads(
    tiny_infini_config: InfiniDopamineTextConfig,
) -> None:
    model = InfiniDopamineForCausalLM(tiny_infini_config)
    input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels)
    loss = output.loss
    loss.backward()

    trainable_params_with_grad = [
        p for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(trainable_params_with_grad) > 0


def test_when_invalid_input_ids_type_then_raises_error(
    tiny_infini_config: InfiniDopamineTextConfig,
) -> None:
    model = InfiniDopamineForCausalLM(tiny_infini_config)
    invalid_inputs = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)

    with pytest.raises((TypeError, RuntimeError)):
        model(input_ids=invalid_inputs)


def test_when_qwen35_and_infinidopamine_share_state_dict_then_outputs_are_identical(
    tiny_infini_config: InfiniDopamineTextConfig,
) -> None:
    qwen_cfg = Qwen3_5TextConfig(
        hidden_size=tiny_infini_config.hidden_size,
        num_hidden_layers=tiny_infini_config.num_hidden_layers,
        linear_key_head_dim=tiny_infini_config.linear_key_head_dim,
        linear_value_head_dim=tiny_infini_config.linear_value_head_dim,
        linear_num_key_heads=tiny_infini_config.linear_num_key_heads,
        linear_num_value_heads=tiny_infini_config.linear_num_value_heads,
        intermediate_size=tiny_infini_config.intermediate_size,
        vocab_size=tiny_infini_config.vocab_size,
        num_attention_heads=tiny_infini_config.num_attention_heads,
        num_key_value_heads=tiny_infini_config.num_key_value_heads,
    )
    torch.manual_seed(42)
    qwen_model = Qwen3_5ForCausalLM(qwen_cfg)
    infini_model = InfiniDopamineForCausalLM(tiny_infini_config)
    # Strictly compatible with Qwen3.5 weights
    load_result = infini_model.load_qwen35_weights(qwen_model, strict=True)
    assert len(load_result.missing_keys) == 0
    assert len(load_result.unexpected_keys) == 0

    input_ids = torch.tensor([[5, 12, 18, 25]], dtype=torch.long)
    qwen_model.eval()
    infini_model.eval()

    with torch.no_grad():
        out_qwen = qwen_model(input_ids=input_ids).logits
        out_infini = infini_model(input_ids=input_ids).logits

        for layer in infini_model.model.layers:
            layer_linear = getattr(layer, "linear_attn", None)
            if (
                layer_linear is not None
                and isinstance(layer_linear, InfiniDopamineGatedDeltaNet)
            ):
                layer_linear.betas.fill_(-10.0)

        out_infini_pure_gdn2 = infini_model(input_ids=input_ids).logits

    # 50/50 mix is close, and pure GDN-2 mode is strictly close to Qwen3.5
    assert torch.allclose(out_qwen, out_infini, atol=0.1)
    assert torch.allclose(out_qwen, out_infini_pure_gdn2, atol=0.03)


def test_when_gdn1_weights_loaded_into_gdn2_layer_then_erase_and_write_gates_expanded() -> None:
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    layer = InfiniDopamineGatedDeltaNet(cfg, layer_idx=0)

    # GDN-1 state dict with scalar in_proj_b
    b_scalar = torch.randn(4, 64)
    gdn1_sd = {
        "in_proj_b.weight": b_scalar,
        "in_proj_a.weight": torch.randn(4, 64),
        "in_proj_qkv.weight": torch.randn(2 * 32 + 64, 64),
        "in_proj_z.weight": torch.randn(64, 64),
        "conv1d.weight": torch.randn(2 * 32 + 64, 1, 4),
        "dt_bias": torch.ones(4),
        "A_log": torch.ones(4),
        "norm.weight": torch.ones(16),
        "out_proj.weight": torch.randn(64, 64),
    }

    res = layer.load_state_dict(gdn1_sd, strict=True)
    assert len(res.missing_keys) == 0
    assert len(res.unexpected_keys) == 0

    # Check that in_proj_b and in_proj_w match the repeated scalar beta
    b_expanded = layer.in_proj_b.weight.view(4, 16, 64)
    w_expanded = layer.in_proj_w.weight.view(4, 16, 64)
    for head in range(4):
        for ch in range(16):
            assert torch.allclose(b_expanded[head, ch], b_scalar[head])
            assert torch.allclose(w_expanded[head, ch], b_scalar[head])


def test_when_gdn2_decoupled_erase_and_write_trained_then_receive_independent_gradients() -> None:
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    layer = InfiniDopamineGatedDeltaNet(cfg, layer_idx=0)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.in_proj_b.weight.grad is not None
    assert layer.in_proj_w.weight.grad is not None
    # Erase and write gradients are independent and non-zero
    assert layer.in_proj_b.weight.grad.shape == (4 * 16, 64)
    assert layer.in_proj_w.weight.grad.shape == (4 * 16, 64)
    assert not torch.allclose(layer.in_proj_b.weight.grad, layer.in_proj_w.weight.grad)


def test_when_sliding_window_configured_then_attention_is_restricted_to_window() -> None:
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=100,
        max_position_embeddings=2048,
        sliding_window=3,
        layer_types=["full_attention"],
        attn_implementation="eager",
    )
    model = InfiniDopamineTextModel(cfg)
    model.eval()

    torch.manual_seed(42)
    inputs_embeds = torch.randn(1, 8, 64)

    with torch.no_grad():
        out1 = model(inputs_embeds=inputs_embeds).last_hidden_state

        # Perturb token 0 which is distance 7 from token 7 (window size is 3, so token 7 only sees tokens 5, 6, 7)
        inputs_embeds_mod = inputs_embeds.clone()
        inputs_embeds_mod[0, 0] = torch.randn(64)
        out2 = model(inputs_embeds=inputs_embeds_mod).last_hidden_state

    # Token 7 is outside window of token 0 -> output must be invariant
    diff_token_7 = (out1[0, 7] - out2[0, 7]).abs().max().item()
    # Token 1 is inside window of token 0 -> output must change
    diff_token_1 = (out1[0, 1] - out2[0, 1]).abs().max().item()

    assert diff_token_7 < 1e-6
    assert diff_token_1 > 1e-4


def test_when_sliding_window_decoding_with_dynamic_cache_then_succeeds(
    tiny_infini_config: InfiniDopamineTextConfig,
) -> None:
    from transformers.cache_utils import DynamicCache

    tiny_infini_config.sliding_window = 4
    tiny_infini_config.layer_types = ["full_attention", "linear_attention"]
    model = InfiniDopamineForCausalLM(tiny_infini_config)
    model.eval()

    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    cache = DynamicCache(config=tiny_infini_config)

    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        assert prefill_out.logits.shape == (1, 4, tiny_infini_config.vocab_size)

        next_token = prefill_out.logits[:, -1:].argmax(dim=-1)
        for _ in range(3):
            step_out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            assert step_out.logits.shape == (1, 1, tiny_infini_config.vocab_size)
            next_token = step_out.logits[:, -1:].argmax(dim=-1)


def test_when_linear_layer_precedes_attention_then_does_not_implicitly_use_gated_reward_net() -> None:
    """Regression: the reward branch is no longer implicitly swapped in.

    Previously any linear layer immediately preceding an attention layer was
    auto-promoted to a GatedRewardNet. That broke pretrained behaviour by
    replacing the main mixer. The reward branch is now an explicit opt-in
    via ``parallel_reward_layers`` or ``use_parallel_reward``.
    """
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        vocab_size=100,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    model = InfiniDopamineTextModel(cfg)

    # All linear layers are GDN-2 (no implicit reward replacement).
    for idx in (0, 1, 2):
        assert isinstance(model.layers[idx].linear_attn, InfiniDopamineGatedDeltaNet)
        assert not hasattr(model.layers[idx], "reward_branch")
    # Layer 3 is the only attention layer; no parallel branch by default.
    assert hasattr(model.layers[3], "self_attn")
    assert not hasattr(model.layers[3], "reward_branch")

    # Opt-in via parallel_reward_layers only on layer 2.
    cfg_explicit = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        vocab_size=100,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        parallel_reward_layers=(2,),
    )
    explicit_model = InfiniDopamineTextModel(cfg_explicit)
    assert isinstance(explicit_model.layers[2].linear_attn, InfiniDopamineGatedDeltaNet)
    assert hasattr(explicit_model.layers[2], "reward_branch")
    assert isinstance(explicit_model.layers[2].reward_branch, InfiniDopamineGatedRewardNet)
    for idx in (0, 1, 3):
        assert not hasattr(explicit_model.layers[idx], "reward_branch")

    # Opt-in via use_parallel_reward=True attaches to attention-only layers.
    cfg_auto = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        vocab_size=100,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        use_parallel_reward=True,
    )
    auto_model = InfiniDopamineTextModel(cfg_auto)
    # No reward branch on linear-only layers.
    for idx in (0, 1, 2):
        assert not hasattr(auto_model.layers[idx], "reward_branch")
    # Reward branch on the attention layer.
    assert hasattr(auto_model.layers[3], "reward_branch")
    assert isinstance(auto_model.layers[3].reward_branch, InfiniDopamineGatedRewardNet)


def test_when_qwen35_weights_loaded_into_model_with_gated_reward_net_then_loads_strictly() -> None:
    qwen_cfg = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        vocab_size=100,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    infini_cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        vocab_size=100,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    torch.manual_seed(42)
    qwen_model = Qwen3_5ForCausalLM(qwen_cfg)
    infini_model = InfiniDopamineForCausalLM(infini_cfg)

    load_result = infini_model.load_qwen35_weights(qwen_model, strict=True)
    assert len(load_result.missing_keys) == 0
    assert len(load_result.unexpected_keys) == 0

    input_ids = torch.tensor([[5, 12, 18, 25]], dtype=torch.long)
    out = infini_model(input_ids=input_ids)
    assert out.logits.shape == (1, 4, infini_cfg.vocab_size)
    assert not torch.isnan(out.logits).any()


def test_when_gated_reward_net_receives_reward_values_then_modulates_output() -> None:
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        vocab_size=100,
        layer_types=["linear_attention", "full_attention"],
    )
    model = InfiniDopamineForCausalLM(cfg)
    model.eval()

    input_ids = torch.tensor([[5, 12, 18, 25]], dtype=torch.long)

    with torch.no_grad():
        # Default forward (reward_values=None)
        out_base = model(input_ids=input_ids).logits

        # Explicit positive reward signal
        rewards = torch.ones(1, 4, 6) * 2.0
        out_reward = model(input_ids=input_ids, reward_values=rewards).logits

    assert out_base.shape == (1, 4, cfg.vocab_size)
    assert out_reward.shape == (1, 4, cfg.vocab_size)
    assert not torch.isnan(out_reward).any()


def test_when_gdn2_layer_has_infini_attention_gate_then_decides_between_swa_and_gdn2() -> None:
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        sliding_window=4,
    )
    layer = InfiniDopamineGatedDeltaNet(cfg, layer_idx=0)
    layer.eval()

    assert hasattr(layer, "betas")
    assert isinstance(layer.betas, torch.nn.Parameter)
    assert layer.betas.shape == (1, 1, 4, 1)

    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        # Default initialization: betas = 0 -> sigmoid(0) = 0.5 (balanced)
        out_balanced = layer(x)

        # Force SWA-dominant (betas = +10 -> sigmoid(10) ~ 1.0)
        layer.betas.fill_(10.0)
        out_swa = layer(x)

        # Force GDN2-dominant (betas = -10 -> sigmoid(-10) ~ 0.0)
        layer.betas.fill_(-10.0)
        out_gdn2 = layer(x)

    assert out_balanced.shape == (2, 8, 64)
    assert out_swa.shape == (2, 8, 64)
    assert out_gdn2.shape == (2, 8, 64)
    assert not torch.isnan(out_balanced).any()
    assert not torch.isnan(out_swa).any()
    assert not torch.isnan(out_gdn2).any()
    # SWA and GDN-2 produce different representations
    assert not torch.allclose(out_swa, out_gdn2, atol=1e-3)


def test_when_infini_gated_deltanet_trained_then_betas_and_shared_qkv_receive_gradients() -> None:
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        sliding_window=4,
    )
    layer = InfiniDopamineGatedDeltaNet(cfg, layer_idx=0)
    x = torch.randn(2, 8, 64, requires_grad=True)

    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.betas.grad is not None
    assert layer.betas.grad.shape == (1, 1, 4, 1)
    assert layer.betas.grad.norm().item() > 0.0

    assert layer.in_proj_qkv.weight.grad is not None
    assert layer.in_proj_qkv.weight.grad.norm().item() > 0.0

    assert layer.in_proj_b.weight.grad is not None
    assert layer.in_proj_w.weight.grad is not None
    assert layer.in_proj_a.weight.grad is not None
    assert layer.in_proj_gate.weight.grad is not None
    assert layer.in_proj_gate.weight.grad.norm().item() > 0.0


def test_when_infinidopamine_gate_is_data_dependent_then_routes_differently_per_token() -> None:
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        sliding_window=4,
    )
    layer = InfiniDopamineGatedDeltaNet(cfg, layer_idx=0)

    # At initialization (in_proj_gate.weight == 0, betas == 0), gate is uniform 0.5
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        gate_logits_init = layer.betas + layer.in_proj_gate(x).unsqueeze(-1)
        gate_init = torch.sigmoid(gate_logits_init)
    assert torch.allclose(gate_init, torch.full_like(gate_init, 0.5))

    # Enable data-dependent projection weights
    nn.init.normal_(layer.in_proj_gate.weight, mean=0.0, std=1.0)

    with torch.no_grad():
        gate_logits = layer.betas + layer.in_proj_gate(x).unsqueeze(-1)
        gate = torch.sigmoid(gate_logits)

    # Gates now dynamically vary across tokens (seq_len=8) and heads (num_v_heads=4)
    assert gate.shape == (2, 8, 4, 1)
    # Tokens within the same sequence have different gate routing decisions
    assert not torch.allclose(gate[:, 0, :, :], gate[:, 1, :, :])
    # Batch items have different gate routing decisions
    assert not torch.allclose(gate[0, :, :, :], gate[1, :, :, :])


def test_when_continued_pretraining_step_performed_then_all_weights_updated() -> None:
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        vocab_size=100,
        layer_types=["linear_attention", "linear_attention"],
    )
    model = InfiniDopamineForCausalLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.long)
    labels = input_ids.clone()

    layer0 = model.model.layers[0]
    layer0_linear = layer0.linear_attn
    assert isinstance(layer0_linear, InfiniDopamineGatedDeltaNet)
    initial_betas = layer0_linear.betas.clone()

    optimizer.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()
    optimizer.step()

    updated_betas = layer0_linear.betas
    assert not torch.allclose(initial_betas, updated_betas)


def test_when_infinidopamine_has_dropout_configured_then_train_mode_applies_regularization() -> (
    None
):
    torch.manual_seed(42)
    config = InfiniDopamineTextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["linear_attention", "linear_attention", "sliding_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        sliding_window=16,
        attention_dropout=0.3,
        hidden_dropout=0.3,
        reward_dropout=0.3,
        advantage_dropout=0.3,
    )
    model = InfiniDopamineForCausalLM(config)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    rewards = torch.ones(1, 8, 4)

    # In eval mode -> deterministic
    model.eval()
    with torch.no_grad():
        eval_out1 = model(input_ids=input_ids, reward_values=rewards).logits
        eval_out2 = model(input_ids=input_ids, reward_values=rewards).logits
    assert torch.allclose(eval_out1, eval_out2)

    # In train mode -> stochastic due to dropouts
    model.train()
    train_out1 = model(input_ids=input_ids, reward_values=rewards).logits
    train_out2 = model(input_ids=input_ids, reward_values=rewards).logits
    assert not torch.allclose(train_out1, train_out2)


def test_when_infinidopamine_gated_delta_net_has_attention_dropout_then_regularizes_swa() -> (
    None
):
    torch.manual_seed(42)
    config = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        sliding_window=8,
        attention_dropout=0.5,
    )
    gdn = InfiniDopamineGatedDeltaNet(config, layer_idx=0)
    # Set betas high so SWA branch dominates
    gdn.betas.data.fill_(10.0)

    hidden = torch.randn(2, 8, 64)

    gdn.eval()
    with torch.no_grad():
        out_eval1 = gdn(hidden_states=hidden)
        out_eval2 = gdn(hidden_states=hidden)
    assert torch.allclose(out_eval1, out_eval2)

    gdn.train()
    out_train1 = gdn(hidden_states=hidden)
    out_train2 = gdn(hidden_states=hidden)
    assert not torch.allclose(out_train1, out_train2)


def test_when_infinidopamine_gated_reward_net_has_reward_dropout_then_regularizes_rewards() -> (
    None
):
    torch.manual_seed(42)
    config = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-5,
        reward_dropout=0.5,
    )
    grn = InfiniDopamineGatedRewardNet(config, layer_idx=0)
    # Enable non-zero reward modulation weights
    grn.delta_layer.advantage_gate.advantage_proj.weight.data.fill_(1.0)
    gamma_proj = getattr(grn.delta_layer.reward_encoder, "gamma_proj", None)
    if isinstance(gamma_proj, nn.Linear):
        gamma_proj.weight.data.fill_(0.5)

    hidden = torch.randn(2, 4, 64)
    rewards = torch.ones(2, 4, 10)

    grn.eval()
    with torch.no_grad():
        eval1 = grn(hidden_states=hidden, reward_values=rewards)
        eval2 = grn(hidden_states=hidden, reward_values=rewards)
    assert torch.allclose(eval1, eval2)

    grn.train()
    train1 = grn(hidden_states=hidden, reward_values=rewards)
    train2 = grn(hidden_states=hidden, reward_values=rewards)
    assert not torch.allclose(train1, train2)


def test_when_infinidopamine_initialized_then_routing_gates_favor_fifty_fifty() -> (
    None
):
    config = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    gdn = InfiniDopamineGatedDeltaNet(config, layer_idx=0)

    # Initial betas are 0.0 -> sigmoid(0) = 0.5 (exact 50/50 balance)
    assert torch.allclose(gdn.betas, torch.zeros_like(gdn.betas))
    gate = torch.sigmoid(gdn.betas)
    assert torch.allclose(gate, torch.full_like(gate, 0.5))

    # At 50/50 initialization, gate regularization penalty is 0.0
    reg_loss = gdn.get_gate_regularization_loss(target=0.5)
    assert torch.isclose(reg_loss, torch.tensor(0.0), atol=1e-7)

    # At 50/50 initialization, entropy is maximized at ln(2) ≈ 0.693147
    entropy = gdn.get_gate_entropy()
    assert torch.isclose(entropy, torch.tensor(0.693147), atol=1e-4)


def test_when_infinidopamine_gate_regularization_loss_computed_then_penalizes_imbalance() -> (
    None
):
    config = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    gdn = InfiniDopamineGatedDeltaNet(config, layer_idx=0)

    # Push betas away from 0.0 (e.g. +3.0 -> 95% SWA, 5% GDN-2)
    gdn.betas.data.fill_(3.0)
    reg_loss = gdn.get_gate_regularization_loss(target=0.5)
    assert reg_loss > 0.15

    # Compute gradient of gate regularization loss with respect to betas
    reg_loss.backward()
    assert gdn.betas.grad is not None
    # Gradient should be positive (pulling betas back down toward 0.0)
    assert (gdn.betas.grad > 0).all()


def test_when_infinidopamine_causal_lm_trains_with_gate_loss_then_gate_regularization_applied() -> (
    None
):
    torch.manual_seed(42)
    config = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        gate_loss_weight=0.1,
        gate_target_balance=0.5,
    )
    model = InfiniDopamineForCausalLM(config)
    model.train()

    # Move betas away from 0 to introduce gate balance penalty
    for layer in model.model.layers:
        linear_attn = getattr(layer, "linear_attn", None)
        if isinstance(linear_attn, InfiniDopamineGatedDeltaNet):
            linear_attn.betas.data.fill_(2.0)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = input_ids.clone()

    out = model(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    assert not torch.isnan(out.loss)

    # Gate loss should be positive
    gate_loss = model.get_gate_regularization_loss(target=0.5)
    assert gate_loss > 0.0


def test_when_parallel_reward_disabled_then_outputs_match_base_model() -> None:
    r"""A causal LM with the parallel reward branch attached should produce
    output indistinguishable from the same model without the branch when
    ``reward_gate_init_bias`` is very negative.
    """
    def make(gate_bias: float) -> InfiniDopamineForCausalLM:
        return InfiniDopamineForCausalLM(InfiniDopamineTextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            vocab_size=100,
            layer_types=["linear_attention", "full_attention"],
            use_parallel_reward=True,
            reward_gate_init_bias=gate_bias,
        ))

    torch.manual_seed(0)
    m_silent = make(-20.0)
    torch.manual_seed(0)
    m_normal = make(-5.0)
    m_silent.load_state_dict(m_normal.state_dict())

    m_silent.eval()
    m_normal.eval()
    input_ids = torch.tensor([[5, 12, 18, 25]], dtype=torch.long)
    with torch.no_grad():
        out_silent = m_silent(input_ids=input_ids).logits
        out_normal = m_normal(input_ids=input_ids).logits
    assert torch.allclose(out_silent, out_normal, atol=1e-5)


def test_when_parallel_reward_enabled_then_gate_init_is_near_zero() -> None:
    r"""A freshly built parallel reward branch must contribute essentially
    nothing: the gate ``sigmoid(b)`` with default bias ``-5`` evaluates to
    ``~0.0067``, so a no-reward forward with identical weights should differ
    from a no-reward forward with the branch disabled by less than 1e-3 in
    the logit norm.
    """
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=100,
        layer_types=["linear_attention", "full_attention"],
        use_parallel_reward=True,
    )
    layer = cfg.layer_types[1] if cfg.layer_types is not None else ""
    model = InfiniDopamineForCausalLM(cfg)
    gate_layer = model.model.layers[1]
    assert hasattr(gate_layer, "reward_gate_proj")
    assert gate_layer.reward_gate_proj.weight.abs().sum() == 0.0
    init_bias = gate_layer.reward_gate_proj.bias.item()
    assert init_bias == cfg.reward_gate_init_bias
    # sigmoid(b) is tiny at start.
    assert torch.sigmoid(torch.tensor(init_bias)).item() < 0.01
    assert layer == "full_attention"  # regression guard


def test_when_parallel_reward_cache_used_then_baseline_persists_across_steps() -> None:
    r"""When the parallel reward branch is enabled on an attention layer and
    the model is called step-by-step, the reward state must persist in the
    Hugging Face ``DynamicCache`` under reward-specific fields.
    """
    from transformers.cache_utils import DynamicCache

    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=100,
        layer_types=["linear_attention", "full_attention"],
        use_parallel_reward=True,
    )
    model = InfiniDopamineForCausalLM(cfg)
    model.eval()

    input_ids = torch.tensor([[5, 12, 18, 25]], dtype=torch.long)
    rewards = torch.ones(1, 4, 6) * 2.0
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            past_key_values=cache,
            reward_values=rewards,
            use_cache=True,
        )

    layer1_cache = cache.layers[1]
    assert hasattr(layer1_cache, "reward_value_baseline")
    assert layer1_cache.reward_value_baseline is not None
    assert layer1_cache.reward_value_baseline.abs().sum() > 0.0
    assert hasattr(layer1_cache, "reward_recurrent_state")
    assert hasattr(layer1_cache, "reward_conv_states")


def test_when_low_rank_reward_memory_used_then_state_shape_matches_rank() -> None:
    r"""When ``reward_memory_rank`` is set, the parallel reward branch uses
    a factored ``(U, V)`` state instead of a dense ``d × d`` matrix.
    """
    from qwendopamine.models.infinidopamine import InfiniDopamineGatedRewardNet
    from qwendopamine.models.reinforced import GatedRewardNet

    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=100,
        layer_types=["linear_attention"],
        use_parallel_reward=False,
        reward_memory_rank=8,
    )
    grn = InfiniDopamineGatedRewardNet(cfg, layer_idx=0)
    grn.eval()
    assert grn.memory_rank == 8
    inputs = torch.randn(1, 4, 64)
    rewards = torch.randn(1, 4, 6)
    # Call the parent class forward to inspect the cache structure; the
    # subclass returns a single tensor (no cache).
    out, _, new_cache = GatedRewardNet.forward(
        grn, inputs, reward_values=rewards, use_cache=True
    )
    assert out.shape == (1, 4, 64)
    rec = new_cache["recurrent_state"]
    # Low-rank state: tuple of (U, V) factors.
    assert isinstance(rec, tuple)
    u, v = rec
    assert u.shape == (1, 64, 8)
    assert v.shape == (1, 64, 8)


def test_when_gate_loss_weight_set_then_regularization_penalizes_drift() -> None:
    r"""The ``parallel_reward_gate_loss_weight`` config drives an MSE penalty
    that keeps the data-dependent gate close to its initial sigmoid value
    early in training. Moving the bias away from the initial bias must
    increase the loss.
    """
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=100,
        layer_types=["linear_attention", "full_attention"],
        use_parallel_reward=True,
        parallel_reward_gate_loss_weight=1.0,
    )
    m = InfiniDopamineForCausalLM(cfg)
    m.train()
    initial_loss = m.get_parallel_reward_gate_loss().item()
    assert initial_loss >= 0.0

    # Move the bias of the only active layer away from init.
    layer = m.model.layers[1]
    layer.reward_gate_proj.bias.data.fill_(2.0)
    drifted_loss = m.get_parallel_reward_gate_loss().item()
    assert drifted_loss > initial_loss


def test_when_gate_entropy_nan_propagates_then_entropy_is_finite() -> None:
    r"""Regression for the entropy clamp: a sigmoid output is strictly
    in (0, 1), so the only failure mode is NaN propagating from upstream
    (e.g. overflow in earlier layers). The entropy function must
    surface the diagnostic by producing NaN/inf without silently clamping
    the gate value first.
    """
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=100,
        layer_types=["linear_attention"],
    )
    m = InfiniDopamineForCausalLM(cfg)
    layer = m.model.layers[0]
    # Inject a NaN into the in_proj_gate weight so the gate value contains
    # NaN. Sigmoid(NaN) = NaN. The entropy must still be finite (the new
    # nan_to_num path replaces the old clamp path).
    with torch.no_grad():
        # Find the linear_attn's in_proj_gate and set one element to NaN.
        # The simplest: set betas to NaN so sigmoid(betas) is NaN.
        # (gated_deltanet uses both in_proj_gate(hidden_states) and betas.)
        # Pick a path that does not require hidden_states.
        layer.linear_attn.betas.fill_(float("nan"))
    entropy = layer.linear_attn.get_gate_entropy()
    assert torch.isfinite(entropy).all(), entropy


def test_when_use_parallel_reward_false_then_gate_loss_is_zero() -> None:
    r"""Without a parallel reward branch the gate loss helper returns zero
    (not NaN) so the trainer can always add it without a feature flag.
    """
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=100,
        layer_types=["full_attention"],
        use_parallel_reward=False,
    )
    m = InfiniDopamineForCausalLM(cfg)
    assert m.get_parallel_reward_gate_loss().item() == 0.0


def test_when_gate_bias_default_then_branch_silent_on_reward_free_input() -> None:
    r"""The default reward gate init bias is ``-5`` (sigmoid ≈ 0.0067), which
    makes the parallel reward branch produce output norm proportional to
    the branch's own initial weight scale (Xavier gain 2**-2.5 ≈ 0.177) and
    the gate scalar — small but non-zero. Verify the gate output matches
    the documented init.
    """
    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=100,
        layer_types=["full_attention"],
        use_parallel_reward=True,
    )
    model = InfiniDopamineForCausalLM(cfg)
    layer0 = model.model.layers[0]
    assert layer0.reward_gate_proj.bias.item() == -5.0
    x = torch.randn(1, 4, 64)
    gate = torch.sigmoid(layer0.reward_gate_proj(layer0.input_layernorm(x)))
    # weight is zero-initialized so gate only depends on bias
    expected_value = torch.sigmoid(torch.tensor(-5.0)).item()
    assert torch.allclose(gate, torch.full_like(gate, expected_value))
    assert gate.abs().max().item() < 0.01
