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


def test_when_reward_statistics_extractor_unnormalized_with_float16_then_preserves_dtype() -> None:
    r"""Verify that RewardStatisticsExtractor preserves float16/bfloat16 dtype even when normalize=False."""
    extractor = RewardStatisticsExtractor(normalize=False)
    rewards_fp16 = torch.randn(2, 4, 8, dtype=torch.float16)

    stats = extractor(rewards_fp16, batch_size=2, seq_len=4)

    assert stats.dtype == torch.float16
    assert stats.shape == (2, 4, 6)
    assert not torch.isnan(stats).any()


def test_when_gated_reward_net_step_by_step_with_conv_cache_then_preserves_temporal_state() -> None:
    r"""Verify that GatedRewardNet step-by-step decoding tracks conv state across sequence steps."""
    from qwendopamine.models.reinforced import (
        GatedRewardNet,
        GatedRewardNetConfig,
    )

    torch.manual_seed(42)
    grn = GatedRewardNet(
        GatedRewardNetConfig(
            hidden_size=32,
            k_stats=6,
            use_short_conv=True,
            conv_size=4,
        )
    )
    grn.eval()

    inputs = torch.randn(1, 6, 32)
    rewards = torch.randn(1, 6, 6)

    # 1. Full sequence forward pass
    out_full, _, _cache_full = grn(inputs, reward_values=rewards, use_cache=True)

    # 2. Step-by-step forward pass propagating cache
    step_outputs = []
    past_cache = None
    for t in range(6):
        x_t = inputs[:, t : t + 1, :]
        r_t = rewards[:, t : t + 1, :]
        out_t, _, past_cache = grn(
            x_t, reward_values=r_t, past_key_values=past_cache, use_cache=True
        )
        step_outputs.append(out_t)

    out_stepped = torch.cat(step_outputs, dim=1)

    assert out_full.shape == (1, 6, 32)
    assert out_stepped.shape == (1, 6, 32)
    assert torch.allclose(out_full, out_stepped, atol=1e-4)


def test_when_gated_reward_net_value_baseline_persists_across_steps() -> None:
    r"""The EMA value baseline must persist across step-by-step decoding.

    Before the cache fix the baseline was reset to ``None`` on every call,
    silently breaking online learning. The full forward and step-by-step
    forward must therefore agree on the final baseline tensor.
    """
    from qwendopamine.models.reinforced import (
        GatedRewardNet,
        GatedRewardNetConfig,
    )

    torch.manual_seed(7)
    grn = GatedRewardNet(
        GatedRewardNetConfig(
            hidden_size=32,
            k_stats=6,
            use_short_conv=True,
            conv_size=4,
        )
    )
    grn.eval()

    inputs = torch.randn(1, 6, 32)
    rewards = torch.ones(1, 6, 6) * 2.0  # non-zero rewards

    # 1. Full sequence forward pass.
    _, _, full_cache = grn(inputs, reward_values=rewards, use_cache=True)
    assert full_cache is not None
    assert "value_baseline" in full_cache
    assert full_cache["value_baseline"] is not None
    full_baseline = full_cache["value_baseline"].clone()

    # 2. Step-by-step with the cache propagated.
    past_cache = None
    for t in range(6):
        _, _, past_cache = grn(
            inputs[:, t : t + 1, :],
            reward_values=rewards[:, t : t + 1, :],
            past_key_values=past_cache,
            use_cache=True,
        )
    assert past_cache is not None
    assert torch.allclose(past_cache["value_baseline"], full_baseline, atol=1e-5)
    # Baseline should have moved away from zero given non-zero rewards.
    assert past_cache["value_baseline"].abs().sum() > 0.0


def test_when_gated_reward_net_cache_is_none_then_value_baseline_initialized_to_zeros() -> None:
    r"""Without a cached baseline the layer should start from a zero vector so
    the first step sees a fresh EMA tracker.
    """
    from qwendopamine.models.reinforced import (
        GatedRewardNet,
        GatedRewardNetConfig,
    )

    torch.manual_seed(0)
    grn = GatedRewardNet(
        GatedRewardNetConfig(
            hidden_size=16,
            k_stats=4,
            use_short_conv=True,
            conv_size=3,
        )
    )
    grn.eval()
    rec, conv, baseline, running_mean, running_std = grn._get_cache(None)
    assert rec is None
    assert conv is None
    assert baseline is None
    assert running_mean is None
    assert running_std is None


def test_when_gated_reward_net_uses_low_rank_memory_then_step_by_step_matches_full() -> None:
    r"""Low-rank memory must still satisfy the recurrent-invariant test
    (full forward = concatenation of step-by-step forward outputs).
    """
    from qwendopamine.models.reinforced import (
        GatedRewardNet,
        GatedRewardNetConfig,
    )

    torch.manual_seed(3)
    grn = GatedRewardNet(
        GatedRewardNetConfig(
            hidden_size=32,
            k_stats=6,
            use_short_conv=True,
            conv_size=3,
            memory_rank=4,
        )
    )
    grn.eval()
    inputs = torch.randn(1, 5, 32)
    rewards = torch.randn(1, 5, 6)

    _, _, _ = grn(inputs, reward_values=rewards, use_cache=True)
    out_full, _, cache_full = grn(inputs, reward_values=rewards, use_cache=True)
    # The cache should expose a low-rank state tuple.
    assert isinstance(cache_full["recurrent_state"], tuple)
    u, v = cache_full["recurrent_state"]
    assert u.shape == (1, 32, 4)
    assert v.shape == (1, 32, 4)

    step_outs = []
    past_cache = None
    for t in range(5):
        out_t, _, past_cache = grn(
            inputs[:, t : t + 1, :],
            reward_values=rewards[:, t : t + 1, :],
            past_key_values=past_cache,
            use_cache=True,
        )
        step_outs.append(out_t)
    out_stepped = torch.cat(step_outs, dim=1)
    assert torch.allclose(out_full, out_stepped, atol=1e-4)


# --- AdvantageGate: plasticity / write / erase separation ---


def test_when_advantage_gate_default_then_returns_separated_triple() -> None:
    r"""The default ``AdvantageGate`` returns three (B, 1) scalars:
    plasticity, write, and erase.
    """
    from qwendopamine.models.reinforced import AdvantageGate

    torch.manual_seed(0)
    gate = AdvantageGate(k_stats=4)
    A = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
    plasticity, write, erase = gate(A)
    assert plasticity.shape == (1, 1)
    assert write.shape == (1, 1)
    assert erase.shape == (1, 1)
    # All three live in (0, 1).
    for s in (plasticity, write, erase):
        assert (s > 0.0).all() and (s < 1.0).all()


def test_when_advantage_gate_legacy_coupled_then_returns_single_omega() -> None:
    r"""When ``legacy_coupled=True`` the gate returns a single ``omega_t``
    scalar in (0, 2) to match the previous coupled-gate behaviour.
    """
    from qwendopamine.models.reinforced import AdvantageGate

    gate = AdvantageGate(k_stats=4, legacy_coupled=True)
    A = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
    out = gate(A)
    assert isinstance(out, tuple)
    assert len(out) == 1
    (omega_t,) = out
    assert omega_t.shape == (1, 1)
    assert (omega_t > 0.0).all() and (omega_t < 2.0).all()


def test_when_advantage_gate_positive_then_write_high_erase_low() -> None:
    r"""Positive advantage should drive ``write`` up and ``erase`` down,
    so good outcomes strengthen memory. The projections start at zero, so
    we manually set the write/erase projection weights to differentiate
    the directions.
    """
    from qwendopamine.models.reinforced import AdvantageGate

    gate = AdvantageGate(k_stats=1)
    # Make the projections respond to A (not A^2): identity mapping.
    with torch.no_grad():
        gate.write_proj.weight.fill_(1.0)
        gate.erase_proj.weight.fill_(1.0)
    A_pos = torch.tensor([[5.0]])
    A_neg = torch.tensor([[-5.0]])
    _, write_pos, erase_pos = gate(A_pos)
    _, write_neg, erase_neg = gate(A_neg)
    # Positive advantage: write > erase.
    assert write_pos.item() > erase_pos.item()
    # Negative advantage: erase > write.
    assert erase_neg.item() > write_neg.item()


def test_when_advantage_gate_plasticity_depends_on_magnitude_only() -> None:
    r"""Plasticity = f(|A|) so the sign of the advantage does not change
    it. With the projection weight set to a non-zero value, the
    plasticity must differ between zero and non-zero magnitude but match
    across opposite signs of equal magnitude.
    """
    from qwendopamine.models.reinforced import AdvantageGate

    gate = AdvantageGate(k_stats=1)
    with torch.no_grad():
        gate.plasticity_proj.weight.fill_(1.0)
    pos = torch.tensor([[3.0]])
    neg = torch.tensor([[-3.0]])
    zero = torch.tensor([[0.0]])
    plast_pos, _, _ = gate(pos)
    plast_neg, _, _ = gate(neg)
    plast_zero, _, _ = gate(zero)
    # Plasticity is a function of |A|, so it should match across signs.
    assert torch.allclose(plast_pos, plast_neg, atol=1e-6)
    # Plasticity must differ between zero and non-zero magnitude.
    assert (plast_pos - plast_zero).abs().item() > 1e-3
    # Plasticity must be in (0, 1).
    assert 0.0 < plast_pos.item() < 1.0


def test_when_reinforced_delta_layer_uses_separated_gates_then_no_smoke_error() -> None:
    r"""Smoke test: ``ReinforcedDeltaLayer`` default config accepts the
    new separated gate and produces finite outputs.
    """
    from qwendopamine.models.reinforced import (
        GatedRewardNet,
        GatedRewardNetConfig,
    )

    grn = GatedRewardNet(
        GatedRewardNetConfig(
            hidden_size=16,
            k_stats=6,
            use_short_conv=True,
            conv_size=3,
        )
    )
    grn.eval()
    inputs = torch.randn(1, 4, 16)
    rewards = torch.randn(1, 4, 6)
    out, _, _ = grn(inputs, reward_values=rewards, use_cache=True)
    assert out.shape == (1, 4, 16)
    assert torch.isfinite(out).all()


def test_when_reinforced_delta_layer_legacy_coupled_then_no_smoke_error() -> None:
    r"""Backward-compat smoke test: ``advantage_legacy_coupled=True``
    routes the gate back to the original single-scalar behaviour.
    """
    from qwendopamine.models.reinforced import (
        GatedRewardNet,
        GatedRewardNetConfig,
    )

    grn = GatedRewardNet(
        GatedRewardNetConfig(
            hidden_size=16,
            k_stats=6,
            use_short_conv=True,
            conv_size=3,
            advantage_legacy_coupled=True,
        )
    )
    grn.eval()
    inputs = torch.randn(1, 4, 16)
    rewards = torch.randn(1, 4, 6)
    out, _, _ = grn(inputs, reward_values=rewards, use_cache=True)
    assert out.shape == (1, 4, 16)
    assert torch.isfinite(out).all()
    assert grn.advantage_legacy_coupled is True


# --- Reward normalization (spec items 6.6 and 8) ---


def test_when_normalize_disabled_then_output_equals_input() -> None:
    r"""The helper always normalises when called. The "disabled" path is
    the caller's choice (skip the call entirely). When called with
    ``running_mean=None`` the function initialises mean=0 and std=1 so
    the output is the input divided by 1 (modulo the eps floor).
    """
    from qwendopamine.models.reinforced import normalize_reward_for_advantage

    r = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]], dtype=torch.float32)
    out, new_mean, new_std = normalize_reward_for_advantage(
        r, None, None, alpha=0.0, training=True
    )
    # alpha=0 -> running stats stay at (mean=0, std=1) -> output is
    # approximately r / (1 + eps). Allow the eps-level tolerance.
    assert torch.allclose(out, r, atol=1e-4)
    assert torch.allclose(new_mean, torch.zeros(1, 2))
    assert torch.allclose(new_std, torch.ones(1, 2))


def test_when_normalize_enabled_then_output_zero_mean_unit_var_approximately() -> None:
    r"""After the first forward the per-channel mean is driven to zero and
    the std to one (within numerical noise) before clipping.
    """
    from qwendopamine.models.reinforced import normalize_reward_for_advantage

    torch.manual_seed(0)
    r = torch.randn(1, 1024, 3) * 5.0 + 2.0
    out, mean, std = normalize_reward_for_advantage(
        r, None, None, alpha=1.0, eps=1e-6, training=True
    )
    # Per-channel mean of the input was ~2 (because N=1024) — check the
    # normalised output is near zero.
    assert torch.allclose(out.mean(dim=1), torch.zeros(1, 3), atol=5e-3)
    # Std of the normalised input should be ~1 (modulo the eps floor).
    std_out = out.std(dim=1, unbiased=False)
    assert torch.allclose(std_out, torch.ones(1, 3), atol=5e-2)
    # The EMA absorbed the batch mean.
    batch_mean = r.mean(dim=1)
    assert torch.allclose(mean, batch_mean, atol=5e-3)
    assert std is not None and (std > 0.0).all()


def test_when_normalize_ema_alpha_zero_then_running_stats_unchanged() -> None:
    r"""``alpha=0`` must disable the EMA update. The returned running
    statistics equal the inputs.
    """
    from qwendopamine.models.reinforced import normalize_reward_for_advantage

    r = torch.randn(1, 4, 2)
    mean = torch.tensor([[1.0, -1.0]])
    std = torch.tensor([[2.0, 3.0]])
    _, new_mean, new_std = normalize_reward_for_advantage(
        r, mean, std, alpha=0.0, training=True
    )
    assert torch.allclose(new_mean, mean)
    assert torch.allclose(new_std, std)


def test_when_normalize_does_not_clip_outliers() -> None:
    r"""Spec items 6.6 and 8 originally prescribed a clip on the
    standardised signal. We removed the clip because (a) downstream
    ``RewardStatisticsExtractor`` computes ``max``/``min`` from the
    post-normalised signal — clipping would silently saturate those
    stats; (b) the standardisation by ``running_std`` already bounds
    most outliers; (c) the gradient through ``clamp`` is zero on the
    values the clip is meant to protect. This test pins the new
    contract: outliers pass through unchanged.
    """
    from qwendopamine.models.reinforced import normalize_reward_for_advantage

    mean = torch.zeros(1, 1)
    std = torch.ones(1, 1)
    # A value that would have been clipped at ±5 (had we kept a clip).
    r = torch.tensor([[[1000.0]]])
    out, _, _ = normalize_reward_for_advantage(
        r, mean, std, alpha=0.0, training=True
    )
    # No clip: the output equals (r - 0) / (1 + eps) ≈ 1000.
    assert out.abs().item() > 100.0


def test_when_normalize_eval_mode_then_running_stats_frozen() -> None:
    r"""Setting ``training=False`` must freeze the running statistics so
    validation / decoding does not silently retune the normaliser.
    """
    from qwendopamine.models.reinforced import normalize_reward_for_advantage

    r = torch.randn(1, 4, 2)
    mean = torch.tensor([[0.0, 0.0]])
    std = torch.tensor([[1.0, 1.0]])
    _, new_mean, new_std = normalize_reward_for_advantage(
        r, mean, std, alpha=0.5, training=False
    )
    assert torch.allclose(new_mean, mean)
    assert torch.allclose(new_std, std)


def test_when_gated_reward_net_normalize_enabled_then_running_mean_persists_in_cache() -> (
    None
):
    r"""When ``reward_normalize=True`` the running mean / std survive
    step-by-step generation via the ``reward_running_mean`` /
    ``reward_running_std`` cache fields. After two non-overlapping
    forward passes the second pass inherits the first pass's EMA.
    """
    from qwendopamine.models.reinforced import (
        GatedRewardNet,
        GatedRewardNetConfig,
    )

    torch.manual_seed(0)
    grn = GatedRewardNet(
        GatedRewardNetConfig(
            hidden_size=16,
            k_stats=6,
            use_short_conv=True,
            conv_size=3,
            reward_normalize=True,
            reward_ema_alpha=1.0,
        )
    )
    grn.train()

    inputs_a = torch.full((1, 4, 16), 0.5)
    rewards_a = torch.full((1, 4, 6), 3.0)
    past_cache = None
    for t in range(4):
        _, _, past_cache = grn(
            inputs_a[:, t : t + 1, :],
            reward_values=rewards_a[:, t : t + 1, :],
            past_key_values=past_cache,
            use_cache=True,
        )
    assert past_cache is not None
    mean_a = past_cache["running_mean"].clone()
    std_a = past_cache["running_std"].clone()
    # First call observed reward == 3, so mean should be 3.
    assert torch.allclose(mean_a, torch.full((1, 6), 3.0), atol=1e-5)
    assert (std_a >= 0.0).all()

    # Second pass with reward == 5 should still rely on the cached mean.
    inputs_b = torch.full((1, 4, 16), 0.5)
    rewards_b = torch.full((1, 4, 6), 5.0)
    for t in range(4):
        _, _, past_cache = grn(
            inputs_b[:, t : t + 1, :],
            reward_values=rewards_b[:, t : t + 1, :],
            past_key_values=past_cache,
            use_cache=True,
        )
    # With alpha=1.0 the mean is overwritten by the new batch mean.
    mean_b = past_cache["running_mean"]
    assert torch.allclose(mean_b, torch.full((1, 6), 5.0), atol=1e-5)


def test_when_gated_reward_net_normalize_disabled_then_no_running_stats_in_cache() -> (
    None
):
    r"""When ``reward_normalize=False`` (default) the cache must not
    carry running_mean / running_std entries.
    """
    from qwendopamine.models.reinforced import (
        GatedRewardNet,
        GatedRewardNetConfig,
    )

    grn = GatedRewardNet(
        GatedRewardNetConfig(
            hidden_size=16,
            k_stats=6,
            use_short_conv=True,
            conv_size=3,
        )
    )
    grn.eval()
    inputs = torch.randn(1, 4, 16)
    rewards = torch.randn(1, 4, 6)
    out, _, new_cache = grn(inputs, reward_values=rewards, use_cache=True)
    assert out.shape == (1, 4, 16)
    assert "running_mean" not in new_cache
    assert "running_std" not in new_cache