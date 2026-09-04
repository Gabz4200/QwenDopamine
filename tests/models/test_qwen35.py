"""Behavioral tests for the Qwen3.5 modular architecture."""

from __future__ import annotations

import pytest
import torch

from qwendopamine.models.qwen35 import (
    Qwen3_5Config,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
    Qwen3_5VisionConfig,
)


def test_when_default_config_instantiated_then_contains_valid_defaults() -> None:
    config = Qwen3_5TextConfig()
    assert config.hidden_size > 0
    assert config.num_hidden_layers > 0
    assert config.vocab_size > 0
    assert config.model_type == "qwen3_5_text"


def test_when_vision_config_instantiated_then_contains_valid_defaults() -> None:
    config = Qwen3_5VisionConfig()
    assert config.out_hidden_size > 0
    assert config.num_position_embeddings > 0


def test_when_multimodal_config_instantiated_then_contains_valid_defaults() -> None:
    config = Qwen3_5Config()
    assert config.image_token_id == 248056
    assert config.vision_start_token_id == 248053
    assert config.vision_end_token_id == 248054


def test_when_text_model_forward_with_mask_then_returns_valid_hidden_states(
    tiny_qwen35_config: Qwen3_5TextConfig,
) -> None:
    model = Qwen3_5TextModel(tiny_qwen35_config)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long)

    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    assert output.last_hidden_state.shape == (2, 4, tiny_qwen35_config.hidden_size)
    assert not torch.isnan(output.last_hidden_state).any()


def test_when_gated_deltanet_forward_then_preserves_shape(
    tiny_qwen35_config: Qwen3_5TextConfig,
) -> None:
    layer = Qwen3_5GatedDeltaNet(tiny_qwen35_config, layer_idx=0)
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, tiny_qwen35_config.hidden_size)

    output = layer(hidden_states)
    assert output.shape == (batch_size, seq_len, tiny_qwen35_config.hidden_size)
    assert not torch.isnan(output).any()


def test_when_decoder_layer_forward_then_preserves_shape(
    tiny_qwen35_config: Qwen3_5TextConfig,
) -> None:
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

    trainable_params_with_grad = [
        p for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(trainable_params_with_grad) > 0


def test_when_invalid_input_ids_type_then_raises_error(
    tiny_qwen35_config: Qwen3_5TextConfig,
) -> None:
    model = Qwen3_5ForCausalLM(tiny_qwen35_config)
    invalid_inputs = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)

    with pytest.raises((TypeError, RuntimeError)):
        model(input_ids=invalid_inputs)


def test_when_left_padded_input_then_unpadded_prefix_matches(
    tiny_qwen35_config: Qwen3_5TextConfig,
) -> None:
    """Left-padded input sequences must produce the same hidden state at the
    first non-pad position as the unpadded counterpart.

    This is the regression test for the review issue 1.6 (left-padded
    conv-state handling on the cached single-step decode path). The
    layer must not leak the pad-id conv kernel state into the first
    real-token update.

    We exercise the Qwen3_5GatedDeltaNet forward path on (a) a
    left-padded input with an attention_mask marking the pad columns,
    and (b) the same input with the pad columns stripped. The first
    real-token output of the padded run must equal the first-token
    output of the unpadded run.
    """
    from qwendopamine.models.qwen35 import Qwen3_5GatedDeltaNet

    config = tiny_qwen35_config
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    layer.eval()
    pad_id = config.vocab_size - 1
    seq_len = 8
    pad = 3
    torch.manual_seed(0)
    real_ids = torch.randint(0, pad_id, (1, seq_len - pad))
    full_ids = torch.cat(
        [torch.full((1, pad), pad_id, dtype=real_ids.dtype), real_ids], dim=1
    )
    # Build hidden states by linear projection through the model's own
    # embedding-free input projection: we don't have an embedding on
    # the bare mixer, so we materialise random embeddings of the right
    # dimensionality.
    embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    with torch.no_grad():
        real_emb = embed(real_ids)
        full_emb = embed(full_ids)

    with torch.no_grad():
        # Unpadded baseline.
        baseline = layer(real_emb, attention_mask=None)
        # Left-padded run with the attention_mask flagging the
        # leading pad positions. The layer is expected to apply
        # ``apply_mask_to_padding_states`` which zeros hidden states
        # at pad positions and lets the recurrence continue from
        # the first real token.
        attention_mask = torch.tensor([[0] * pad + [1] * (seq_len - pad)])
        padded = layer(full_emb, attention_mask=attention_mask)

    assert torch.allclose(padded[0, pad], baseline[0, 0], atol=1e-5), (
        "Left-padded run diverged from unpadded baseline at first real token"
    )
