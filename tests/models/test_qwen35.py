"""Behavioral tests for the Qwen3.5 modular architecture."""

from __future__ import annotations

import pytest
import torch

from qwendopamine.models.qwen35 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
)


@pytest.fixture
def tiny_qwen35_config() -> Qwen3_5TextConfig:
    r"""Fixture providing a fast, minimal Qwen3.5 configuration."""
    return Qwen3_5TextConfig(
        hidden_size=64,
        num_hidden_layers=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        intermediate_size=128,
        vocab_size=500,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


def test_when_config_created_then_defaults_are_valid(tiny_qwen35_config: Qwen3_5TextConfig) -> None:
    assert tiny_qwen35_config.hidden_size == 64
    assert tiny_qwen35_config.num_hidden_layers == 4
    assert tiny_qwen35_config.vocab_size == 500
    assert isinstance(tiny_qwen35_config.mlp_only_layers, list)
    assert tiny_qwen35_config.decoder_sparse_step == 1


def test_when_gated_deltanet_forward_then_preserves_shape(tiny_qwen35_config: Qwen3_5TextConfig) -> None:
    layer = Qwen3_5GatedDeltaNet(tiny_qwen35_config, layer_idx=0)
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, tiny_qwen35_config.hidden_size)

    output = layer(hidden_states)
    assert output.shape == (batch_size, seq_len, tiny_qwen35_config.hidden_size)
    assert not torch.isnan(output).any()


def test_when_decoder_layer_forward_then_preserves_shape(tiny_qwen35_config: Qwen3_5TextConfig) -> None:
    layer = Qwen3_5DecoderLayer(tiny_qwen35_config, layer_idx=0)
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, tiny_qwen35_config.hidden_size)

    output = layer(hidden_states)
    tensor_output = output[0] if isinstance(output, tuple) else output
    assert tensor_output.shape == (batch_size, seq_len, tiny_qwen35_config.hidden_size)
    assert not torch.isnan(tensor_output).any()


def test_when_text_model_forward_then_returns_valid_hidden_states(
    tiny_qwen35_config: Qwen3_5TextConfig,
) -> None:
    model = Qwen3_5TextModel(tiny_qwen35_config)
    input_ids = torch.tensor([[1, 5, 10, 20], [2, 4, 8, 16]], dtype=torch.long)

    output = model(input_ids=input_ids)
    assert output.last_hidden_state.shape == (2, 4, tiny_qwen35_config.hidden_size)
    assert not torch.isnan(output.last_hidden_state).any()


def test_when_causal_lm_forward_then_computes_logits_and_loss(
    tiny_qwen35_config: Qwen3_5TextConfig,
) -> None:
    model = Qwen3_5ForCausalLM(tiny_qwen35_config)
    input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.long)
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels)
    assert output.logits.shape == (2, 4, tiny_qwen35_config.vocab_size)
    assert output.loss is not None
    assert output.loss.item() > 0.0


def test_when_gradients_computed_then_parameters_receive_grads(
    tiny_qwen35_config: Qwen3_5TextConfig,
) -> None:
    model = Qwen3_5ForCausalLM(tiny_qwen35_config)
    input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels)
    output.loss.backward()

    trainable_params_with_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(trainable_params_with_grad) > 0


def test_when_invalid_input_ids_type_then_raises_error(tiny_qwen35_config: Qwen3_5TextConfig) -> None:
    model = Qwen3_5ForCausalLM(tiny_qwen35_config)
    invalid_inputs = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)

    with pytest.raises((TypeError, RuntimeError, ValueError)):
        model(input_ids=invalid_inputs)
