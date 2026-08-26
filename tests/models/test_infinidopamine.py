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
        _attn_implementation="eager",
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


def test_when_linear_layer_precedes_attention_then_uses_gated_reward_net() -> None:
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

    # Layer 0: followed by linear_attention -> GDN-2
    assert isinstance(model.layers[0].linear_attn, InfiniDopamineGatedDeltaNet)
    # Layer 1: followed by linear_attention -> GDN-2
    assert isinstance(model.layers[1].linear_attn, InfiniDopamineGatedDeltaNet)
    # Layer 2: followed by full_attention -> InfiniDopamineGatedRewardNet
    assert isinstance(model.layers[2].linear_attn, InfiniDopamineGatedRewardNet)
    # Layer 3: full_attention -> Attention
    assert hasattr(model.layers[3], "self_attn")


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
