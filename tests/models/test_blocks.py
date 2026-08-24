"""Behavioral tests for model blocks, registry, and model factory."""

from __future__ import annotations

import types

import pytest
import torch

from qwendopamine.models.blocks import (
    BLOCKS,
    GatedDeltaNetBlock,
    LearnableFourierFeatures,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    QwenDecoderLayer,
    RewardFiLM,
    RewardFourierEncoder,
    RewardStatisticsExtractor,
    TokenWiseFiLM,
    build_block,
)
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
    model = build_model(cfg)
    assert isinstance(model, Qwen3_5ForCausalLM)


def test_when_build_model_with_research_config_then_returns_research_decoder(
    mock_config: types.SimpleNamespace,
) -> None:
    model = build_model(mock_config)
    assert isinstance(model, ResearchDecoder)


def test_when_research_decoder_forward_with_gdn2_block_then_executes_successfully(
    mock_config: types.SimpleNamespace,
) -> None:
    mock_config.block_types = ["gdn2"]
    mock_config.hidden_size = 64
    mock_config.num_heads = 2
    mock_config.head_dim = 32
    model = ResearchDecoder(mock_config)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits = model(input_ids)
    assert logits.shape == (2, 3, mock_config.vocab_size)
    assert not torch.isnan(logits).any()


def test_when_token_wise_film_forward_called_then_modulates_features_correctly() -> (
    None
):
    film = TokenWiseFiLM(dim=32)
    x = torch.randn(2, 4, 32)
    cond = torch.randn(2, 64)
    out = film(x, cond)
    assert out.shape == (2, 4, 32)
    assert not torch.isnan(out).any()

    cond_3d = torch.randn(2, 4, 64)
    out_3d = film(x, cond_3d)
    assert out_3d.shape == (2, 4, 32)


def test_when_learnable_fourier_features_forward_then_encodes_position() -> None:
    lff = LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64, g_dim=1)
    pos = torch.randn(2, 5, 1, 4)
    enc = lff(pos)
    assert enc.shape == (2, 5, 64)
    assert not torch.isnan(enc).any()


def test_when_reward_components_forward_called_then_returns_conditioned_features() -> None:
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=64)
    film = RewardFiLM(dim=32, hidden_dim=64)

    x = torch.randn(2, 5, 32)
    reward_2d = torch.randn(2, 5)

    stats = extractor(reward_2d, batch_size=2, seq_len=5)
    cond = fourier(stats)
    out = film(x, cond)
    assert out.shape == (2, 5, 64)
    assert not torch.isnan(out).any()

    reward_3d = torch.randn(2, 5, 3)
    stats_3d = extractor(reward_3d, batch_size=2, seq_len=5)
    cond_3d = fourier(stats_3d)
    out_3d = film(x, cond_3d)
    assert out_3d.shape == (2, 5, 64)
    assert not torch.isnan(out_3d).any()


def test_when_reward_components_dtype_differs_then_aligns_and_executes() -> None:
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=32)
    film = RewardFiLM(dim=32, hidden_dim=32)

    x = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    reward_values = torch.randn(2, 4, 2, dtype=torch.float32)

    extractor.to(dtype=torch.bfloat16)
    fourier.to(dtype=torch.bfloat16)
    film.to(dtype=torch.bfloat16)

    stats = extractor(reward_values, batch_size=2, seq_len=4)
    cond = fourier(stats)
    out = film(x, cond)
    assert out.shape == (2, 4, 32)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()


def test_when_blocks_registry_contains_reward_blocks() -> None:
    assert "reward_stats_extractor" in BLOCKS
    assert BLOCKS["reward_stats_extractor"] is RewardStatisticsExtractor
    assert "reward_fourier_encoder" in BLOCKS
    assert BLOCKS["reward_fourier_encoder"] is RewardFourierEncoder
    assert "reward_film" in BLOCKS
    assert BLOCKS["reward_film"] is RewardFiLM
