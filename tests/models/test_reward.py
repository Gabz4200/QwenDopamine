"""Behavioral tests for reward encoder, AdaLN, and Fourier feature blocks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwendopamine.models.blocks.reward import (
    AsinhScaler,
    LearnableFourierFeatures,
    RewardFiLM,
    RewardFourierEncoder,
    RewardStatisticsExtractor,
    TokenWiseFiLM,
)

# --- TokenWiseFiLM Tests ---


def test_when_token_wise_film_initialized_then_has_correct_dimension() -> None:
    film = TokenWiseFiLM(dim=64)
    assert film.dim == 64
    assert film.cond_dim == 64


def test_when_token_wise_film_forward_with_unit_scale_and_zero_shift_then_matches_modulation() -> (
    None
):
    dim = 16
    film = TokenWiseFiLM(dim=dim)
    x = torch.randn(2, 5, dim)
    cond = torch.zeros(2, 5, dim)
    output = film(x, cond)
    assert output.shape == (2, 5, dim)
    assert torch.allclose(output, x, atol=1e-6)


def test_when_token_wise_film_receives_2d_cond_then_broadcasts_over_sequence_dimension() -> (
    None
):
    dim = 16
    film = TokenWiseFiLM(dim=dim)
    x = torch.randn(2, 5, dim)
    cond = torch.ones(2, dim)  # (B, D) -> should broadcast to (B, L, D)
    output = film(x, cond)
    assert output.shape == (2, 5, dim)
    expected = x * torch.ones_like(x)
    assert torch.allclose(output, expected, atol=1e-6)


def test_when_token_wise_film_forward_backward_then_computes_gradients_for_x_and_cond() -> (
    None
):
    dim = 16
    film = TokenWiseFiLM(dim=dim)
    x = torch.randn(2, 5, dim, requires_grad=True)
    cond = torch.randn(2, 5, dim, requires_grad=True)
    output = film(x, cond)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert cond.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(cond.grad).any()


# --- LearnableFourierFeatures Tests ---


def test_when_learnable_fourier_features_initialized_with_odd_f_dim_then_raises_value_error() -> (
    None
):
    with pytest.raises(ValueError, match="divisible by 2"):
        LearnableFourierFeatures(pos_dim=4, f_dim=15, h_dim=32, d_dim=64)


def test_when_learnable_fourier_features_initialized_with_incompatible_g_dim_then_raises_value_error() -> (
    None
):
    with pytest.raises(ValueError, match="divisible by g_dim"):
        LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64, g_dim=5)


def test_when_learnable_fourier_features_forward_with_include_input_true_then_outputs_expected_shape() -> (
    None
):
    lff = LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64, g_dim=1)
    pos = torch.randn(2, 5, 1, 4)
    out = lff(pos)
    assert out.shape == (2, 5, 64)
    assert not torch.isnan(out).any()


def test_when_learnable_fourier_features_forward_with_include_input_false_then_outputs_expected_shape() -> (
    None
):
    lff = LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64, g_dim=1, include_input=False)
    pos = torch.randn(2, 5, 1, 4)
    out = lff(pos)
    assert out.shape == (2, 5, 64)
    assert not torch.isnan(out).any()


def test_when_learnable_fourier_features_backward_then_updates_parameters() -> None:
    lff = LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64, g_dim=1)
    pos = torch.randn(2, 5, 1, 4, requires_grad=True)
    out = lff(pos)
    loss = out.sum()
    loss.backward()
    for param in lff.parameters():
        assert param.grad is not None


# --- AsinhScaler Tests ---


def test_when_asinh_scaler_forward_called_then_scales_heavy_tailed_features() -> None:
    scaler = AsinhScaler(dim=16)
    x = torch.randn(2, 5, 16) * 100  # large values
    out = scaler(x)
    assert out.shape == (2, 5, 16)
    assert (out.abs() < x.abs()).all()


# --- RewardStatisticsExtractor Tests ---


def test_when_reward_statistics_extractor_forward_then_outputs_six_stats() -> None:
    extractor = RewardStatisticsExtractor()
    rewards = torch.randn(2, 5, 10)
    stats = extractor(rewards, batch_size=2, seq_len=5)
    assert stats.shape == (2, 5, 6)
    assert not torch.isnan(stats).any()


def test_when_reward_statistics_extractor_forward_with_2d_rewards_then_processes_correctly() -> None:
    extractor = RewardStatisticsExtractor()
    rewards = torch.tensor([[0.5, 0.2, 0.9, 0.1, 0.4], [0.3, 0.7, 0.1, 0.8, 0.5]])
    stats = extractor(rewards, batch_size=2, seq_len=5)
    assert stats.shape == (2, 5, 6)
    assert not torch.isnan(stats).any()


def test_when_reward_statistics_extractor_forward_with_3d_multi_rewards_then_computes_stats_correctly() -> (
    None
):
    extractor = RewardStatisticsExtractor()
    rewards = torch.randn(2, 4, 3)
    stats = extractor(rewards, batch_size=2, seq_len=4)
    assert stats.shape == (2, 4, 6)
    assert not torch.isnan(stats).any()


def test_when_reward_statistics_extractor_forward_with_constant_rewards_then_produces_valid_output() -> (
    None
):
    extractor = RewardStatisticsExtractor()
    rewards = torch.ones(2, 3, 4)  # All reward values equal to 1.0
    stats = extractor(rewards, batch_size=2, seq_len=3)
    assert stats.shape == (2, 3, 6)
    assert not torch.isnan(stats).any()


def test_when_reward_statistics_extractor_device_or_dtype_mismatched_then_auto_aligns() -> None:
    extractor = RewardStatisticsExtractor()
    rewards = torch.randn(2, 3, 2, dtype=torch.float32)
    extractor.to(dtype=torch.bfloat16)
    stats = extractor(rewards, batch_size=2, seq_len=3)
    assert stats.shape == (2, 3, 6)
    assert stats.dtype == torch.bfloat16
    assert not torch.isnan(stats).any()


# --- RewardFourierEncoder Tests ---


def test_when_reward_fourier_encoder_initialized_then_has_correct_dimensions() -> None:
    encoder = RewardFourierEncoder(f_dim=32, h_dim=64, d_dim=64)
    assert encoder.f_dim == 32
    assert encoder.h_dim == 64
    assert encoder.d_dim == 64
    assert encoder.g_dim == 1
    assert encoder.include_input is True


def test_when_reward_fourier_encoder_forward_then_outputs_expected_shape() -> None:
    encoder = RewardFourierEncoder(f_dim=32, h_dim=64, d_dim=64)
    stats = torch.randn(2, 5, 6)
    cond = encoder(stats)
    assert cond.shape == (2, 5, 64)
    assert not torch.isnan(cond).any()


def test_when_reward_fourier_encoder_forward_with_custom_d_dim_then_outputs_correct_shape() -> None:
    encoder = RewardFourierEncoder(f_dim=32, h_dim=64, d_dim=128)
    stats = torch.randn(2, 5, 6)
    cond = encoder(stats)
    assert cond.shape == (2, 5, 128)
    assert not torch.isnan(cond).any()


def test_when_reward_fourier_encoder_backward_then_updates_parameters() -> None:
    encoder = RewardFourierEncoder(f_dim=32, h_dim=64, d_dim=64)
    stats = torch.randn(2, 5, 6, requires_grad=True)
    cond = encoder(stats)
    loss = cond.sum()
    loss.backward()
    for param in encoder.parameters():
        assert param.grad is not None


# --- RewardFiLM Tests ---


def test_when_reward_film_initialized_then_has_correct_dimensions() -> None:
    film = RewardFiLM(dim=32, hidden_dim=64)
    assert film.dim == 32
    assert film.hidden_dim == 64


def test_when_reward_film_same_dim_then_uses_identity_projection() -> None:
    film = RewardFiLM(dim=32, hidden_dim=32)
    assert isinstance(film.x_proj, nn.Identity)


def test_when_reward_film_different_dim_then_uses_linear_projection() -> None:
    film = RewardFiLM(dim=32, hidden_dim=64)
    assert isinstance(film.x_proj, nn.Linear)
    assert film.x_proj.in_features == 32
    assert film.x_proj.out_features == 64


def test_when_reward_film_forward_with_3d_input_then_outputs_correct_shape() -> None:
    film = RewardFiLM(dim=32, hidden_dim=32)
    x = torch.randn(2, 5, 32)
    cond = torch.randn(2, 5, 32)
    output = film(x, cond)
    assert output.shape == (2, 5, 32)
    assert not torch.isnan(output).any()


def test_when_reward_film_forward_with_2d_input_then_outputs_correct_shape() -> None:
    film = RewardFiLM(dim=32, hidden_dim=32)
    x = torch.randn(4, 32)  # (B, D)
    cond = torch.randn(4, 32)  # (B, D)
    output = film(x, cond)
    assert output.shape == (4, 32)
    assert not torch.isnan(output).any()


def test_when_reward_film_forward_with_1d_input_then_outputs_correct_shape() -> None:
    film = RewardFiLM(dim=32, hidden_dim=32)
    x = torch.randn(32)  # (D,)
    cond = torch.randn(32)  # (D,)
    output = film(x, cond)
    assert output.shape == (32,)
    assert not torch.isnan(output).any()


def test_when_reward_film_device_or_dtype_mismatched_then_auto_aligns() -> None:
    film = RewardFiLM(dim=16, hidden_dim=16)
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    cond = torch.randn(2, 3, 16, dtype=torch.float32)
    film.to(dtype=torch.bfloat16)
    output = film(x, cond)
    assert output.shape == (2, 3, 16)
    assert output.dtype == torch.bfloat16
    assert not torch.isnan(output).any()


def test_when_reward_film_backward_called_then_gradients_flow_to_inputs_and_params() -> None:
    film = RewardFiLM(dim=16, hidden_dim=16)
    x = torch.randn(2, 3, 16, requires_grad=True)
    cond = torch.randn(2, 3, 16, requires_grad=True)
    output = film(x, cond)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert cond.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(cond.grad).any()
    for param in film.parameters():
        if param.requires_grad:
            assert param.grad is not None


# --- Chained Reward Encoder Tests (Integration) ---


def test_when_chained_reward_components_then_produces_expected_output() -> None:
    """Test that the three components chain together correctly."""
    batch_size, seq_len, dim, hidden_dim = 2, 5, 32, 64
    x = torch.randn(batch_size, seq_len, dim)
    reward_values = torch.randn(batch_size, seq_len, 10)

    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=hidden_dim)
    film = RewardFiLM(dim=dim, hidden_dim=hidden_dim)

    stats = extractor(reward_values, batch_size=batch_size, seq_len=seq_len)
    cond = fourier(stats)
    out = film(x, cond)

    assert out.shape == (batch_size, seq_len, hidden_dim)
    assert not torch.isnan(out).any()


def test_when_chained_reward_components_with_2d_rewards_then_processes_correctly() -> None:
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=32)
    film = RewardFiLM(dim=32, hidden_dim=32)

    x = torch.randn(2, 5, 32)
    reward_2d = torch.tensor([[0.5, 0.2, 0.9, 0.1, 0.4], [0.3, 0.7, 0.1, 0.8, 0.5]])

    stats = extractor(reward_2d, batch_size=2, seq_len=5)
    cond = fourier(stats)
    output = film(x, cond)

    assert output.shape == (2, 5, 32)
    assert not torch.isnan(output).any()


def test_when_chained_reward_components_with_3d_multi_rewards_then_computes_correctly() -> None:
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=32)
    film = RewardFiLM(dim=32, hidden_dim=32)

    x = torch.randn(2, 4, 32)
    reward_3d = torch.randn(2, 4, 3)

    stats = extractor(reward_3d, batch_size=2, seq_len=4)
    cond = fourier(stats)
    output = film(x, cond)

    assert output.shape == (2, 4, 32)
    assert not torch.isnan(output).any()


def test_when_chained_reward_components_with_constant_rewards_then_produces_valid_output() -> None:
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=16)
    film = RewardFiLM(dim=16, hidden_dim=16)

    x = torch.randn(2, 3, 16)
    reward_const = torch.ones(2, 3, 4)

    stats = extractor(reward_const, batch_size=2, seq_len=3)
    cond = fourier(stats)
    output = film(x, cond)

    assert output.shape == (2, 3, 16)
    assert not torch.isnan(output).any()


def test_when_chained_reward_components_device_or_dtype_mismatched_then_auto_aligns() -> None:
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=16)
    film = RewardFiLM(dim=16, hidden_dim=16)

    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    reward_values = torch.randn(2, 3, 2, dtype=torch.float32)

    extractor.to(dtype=torch.bfloat16)
    fourier.to(dtype=torch.bfloat16)
    film.to(dtype=torch.bfloat16)

    stats = extractor(reward_values, batch_size=2, seq_len=3)
    cond = fourier(stats)
    output = film(x, cond)

    assert output.shape == (2, 3, 16)
    assert output.dtype == torch.bfloat16
    assert not torch.isnan(output).any()


def test_when_chained_reward_components_backward_called_then_gradients_flow() -> None:
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=16)
    film = RewardFiLM(dim=16, hidden_dim=16)

    x = torch.randn(2, 3, 16, requires_grad=True)
    reward_values = torch.randn(2, 3, 2, requires_grad=True)

    stats = extractor(reward_values, batch_size=2, seq_len=3)
    cond = fourier(stats)
    output = film(x, cond)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert reward_values.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(reward_values.grad).any()
    for param in extractor.parameters():
        if param.requires_grad:
            assert param.grad is not None
    for param in fourier.parameters():
        if param.requires_grad:
            assert param.grad is not None
    for param in film.parameters():
        if param.requires_grad:
            assert param.grad is not None


@pytest.mark.parametrize("num_rewards", [1, 2, 5, 20, 100])
def test_when_chained_reward_components_receive_unbounded_k_rewards_then_regularizes_and_encodes(
    num_rewards: int,
) -> None:
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=64)
    film = RewardFiLM(dim=32, hidden_dim=64)

    x = torch.randn(2, 8, 32)
    rewards = torch.randn(2, 8, num_rewards)

    stats = extractor(rewards, batch_size=2, seq_len=8)
    cond = fourier(stats)
    out = film(x, cond)

    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("seq_len", [1, 5, 32, 128])
def test_when_chained_reward_components_receive_variable_sequence_lengths_then_executes_successfully(
    seq_len: int,
) -> None:
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=64)
    film = RewardFiLM(dim=32, hidden_dim=64)

    x = torch.randn(2, seq_len, 32)
    rewards = torch.randn(2, seq_len, 4)

    stats = extractor(rewards, batch_size=2, seq_len=seq_len)
    cond = fourier(stats)
    out = film(x, cond)

    assert out.shape == (2, seq_len, 64)
    assert not torch.isnan(out).any()


def test_when_chained_reward_components_receive_single_unbatched_embedding_then_modulates_correctly() -> (
    None
):
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=64)
    film = RewardFiLM(dim=32, hidden_dim=64)

    x_single = torch.randn(32)
    rewards_single = torch.randn(5)

    stats = extractor(rewards_single, batch_size=1, seq_len=1)
    cond = fourier(stats)
    out = film(x_single, cond)

    assert out.shape == (64,)
    assert not torch.isnan(out).any()


def test_when_chained_reward_components_receive_single_step_batched_embeddings_then_modulates_correctly() -> (
    None
):
    extractor = RewardStatisticsExtractor()
    fourier = RewardFourierEncoder(d_dim=64)
    film = RewardFiLM(dim=32, hidden_dim=64)

    x_batched = torch.randn(4, 32)
    rewards_batched = torch.randn(4, 8)

    stats = extractor(rewards_batched, batch_size=4, seq_len=1)
    cond = fourier(stats)
    out = film(x_batched, cond)

    assert out.shape == (4, 64)
    assert not torch.isnan(out).any()


def test_when_reward_statistics_extractor_invalid_dropout_then_raises_error() -> None:
    with pytest.raises(ValueError, match="reward_dropout"):
        RewardStatisticsExtractor(reward_dropout=-0.1)
    with pytest.raises(ValueError, match="reward_dropout"):
        RewardStatisticsExtractor(reward_dropout=1.0)


def test_when_reward_statistics_extractor_dropout_in_train_mode_then_randomly_zeros_rewards() -> (
    None
):
    torch.manual_seed(42)
    extractor = RewardStatisticsExtractor(reward_dropout=0.5)
    extractor.train()

    # Pass a multi-channel reward tensor with all positive 1.0s
    rewards = torch.ones(4, 10, 20)
    stats1 = extractor(rewards, batch_size=4, seq_len=10)
    stats2 = extractor(rewards, batch_size=4, seq_len=10)

    # In training mode with dropout=0.5, stochastic drop causes varying outputs across runs
    assert not torch.allclose(stats1, stats2)
    assert stats1.shape == (4, 10, 6)
    assert not torch.isnan(stats1).any()


def test_when_reward_statistics_extractor_dropout_in_eval_mode_then_deterministic() -> (
    None
):
    extractor = RewardStatisticsExtractor(reward_dropout=0.5)
    extractor.eval()

    rewards = torch.randn(2, 5, 10)
    stats1 = extractor(rewards, batch_size=2, seq_len=5)
    stats2 = extractor(rewards, batch_size=2, seq_len=5)

    # In eval mode, dropout is deactivated
    assert torch.allclose(stats1, stats2)


def test_when_learnable_fourier_features_invalid_dropout_then_raises_error() -> None:
    with pytest.raises(ValueError, match="dropout"):
        LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64, dropout=-0.2)
    with pytest.raises(ValueError, match="dropout"):
        LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64, dropout=1.5)


def test_when_learnable_fourier_features_dropout_in_train_vs_eval_mode() -> None:
    torch.manual_seed(42)
    lff = LearnableFourierFeatures(
        pos_dim=4, f_dim=16, h_dim=32, d_dim=64, dropout=0.5
    )
    pos = torch.randn(2, 5, 1, 4)

    lff.train()
    out1 = lff(pos)
    out2 = lff(pos)
    assert not torch.allclose(out1, out2)

    lff.eval()
    eval1 = lff(pos)
    eval2 = lff(pos)
    assert torch.allclose(eval1, eval2)


def test_when_token_wise_film_dropout_in_train_mode_then_regularizes_conditioning() -> None:
    torch.manual_seed(42)
    film = TokenWiseFiLM(dim=16, dropout=0.5, identity_init=False)
    x = torch.randn(2, 5, 16)
    cond = torch.randn(2, 5, 16)

    film.train()
    out1 = film(x, cond)
    out2 = film(x, cond)
    assert not torch.allclose(out1, out2)

    film.eval()
    eval1 = film(x, cond)
    eval2 = film(x, cond)
    assert torch.allclose(eval1, eval2)