"""Behavioral tests for model blocks, registry, and model factory."""

from __future__ import annotations

import types

import pytest
import torch

from qwendopamine.blocks import (
    GatedDeltaNetBlock,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    QwenDecoderLayer,
)
from qwendopamine.models.blocks import BLOCKS, build_block
from qwendopamine.models.model_factory import (
    ResearchDecoder,
    build_model,
)
from qwendopamine.models.qwen35 import Qwen3_5ForCausalLM, Qwen3_5TextConfig


@pytest.fixture
def mock_config() -> types.SimpleNamespace:
    r"""Fixture providing a mock configuration for research decoder."""
    return types.SimpleNamespace(
        hidden_size=64,
        vocab_size=200,
        max_position_embeddings=128,
        num_layers=2,
        block_types=["qwen", "gdn"],
        layer_types=["linear_attention", "linear_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )


def test_when_blocks_registry_queried_then_contains_qwen35_blocks() -> None:
    assert "qwen" in BLOCKS
    assert "gdn" in BLOCKS
    assert "qwen35" in BLOCKS
    assert "qwen35_gdn" in BLOCKS
    assert BLOCKS["qwen"] is Qwen3_5DecoderLayer
    assert BLOCKS["gdn"] is Qwen3_5GatedDeltaNet


def test_when_build_block_called_then_instantiates_correct_block(
    mock_config: types.SimpleNamespace,
) -> None:
    qwen_layer = build_block("qwen", mock_config, layer_idx=0)
    gdn_layer = build_block("gdn", mock_config, layer_idx=1)

    assert isinstance(qwen_layer, QwenDecoderLayer)
    assert isinstance(gdn_layer, GatedDeltaNetBlock)


def test_when_unknown_block_requested_then_raises_key_error(
    mock_config: types.SimpleNamespace,
) -> None:
    with pytest.raises(KeyError, match="Unknown block type"):
        build_block("nonexistent_block_type", mock_config, layer_idx=0)


def test_when_research_decoder_forward_then_returns_correct_logits(
    mock_config: types.SimpleNamespace,
) -> None:
    model = ResearchDecoder(mock_config)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits = model(input_ids)
    assert logits.shape == (2, 3, mock_config.vocab_size)
    assert not torch.isnan(logits).any()


def test_when_build_model_with_qwen35_config_then_returns_causal_lm() -> None:
    cfg = Qwen3_5TextConfig(
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
    model = build_model(cfg)
    assert isinstance(model, Qwen3_5ForCausalLM)


def test_when_build_model_with_research_config_then_returns_research_decoder(
    mock_config: types.SimpleNamespace,
) -> None:
    model = build_model(mock_config)
    assert isinstance(model, ResearchDecoder)
