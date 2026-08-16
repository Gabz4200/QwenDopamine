"""Behavioral tests for reward encoder, AdaLN, and Fourier feature blocks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwendopamine.models.blocks.reward import (
    AdaLN,
    FourierFeatures,
    LearnableFourierFeatures,
    PositionalEncoding,
    RewardEncoder,
)

# --- AdaLN Tests ---


def test_when_adaln_initialized_then_has_correct_norm_dimension() -> None:
    adaln = AdaLN(dim=64, eps=1e-5)
    assert adaln.dim == 64
    assert isinstance(adaln.norm, nn.RMSNorm)
    assert adaln.norm.eps == 1e-5


def test_when_adaln_forward_with_unit_scale_and_zero_shift_then_matches_rmsnorm() -> None:
    dim = 16
    adaln = AdaLN(dim=dim)
    x = torch.randn(2, 4, dim)

    # cond contains [gamma, beta] concatenated along dim=-1
    # gamma = 1.0, beta = 0.0
    gamma = torch.ones(2, 4, dim)
    beta = torch.zeros(2, 4, dim)
    cond = torch.cat([gamma, beta], dim=-1)

    output = adaln(x, cond)
    expected = adaln.norm(x)

    torch.testing.assert_close(output, expected)


def test_when_adaln_forward_with_scale_and_shift_then_applies_linear_modulation() -> None:
    dim = 8
    adaln = AdaLN(dim=dim)
    x = torch.randn(2, 3, dim)

    scale = torch.full((2, 3, dim), 2.0)
    shift = torch.full((2, 3, dim), 3.0)
    cond = torch.cat([scale, shift], dim=-1)

    output = adaln(x, cond)
    expected = adaln.norm(x) * 2.0 + 3.0

    torch.testing.assert_close(output, expected)


def test_when_adaln_receives_2d_cond_then_broadcasts_over_sequence_dimension() -> None:
    dim = 16
    adaln = AdaLN(dim=dim)
    x = torch.randn(3, 5, dim)
    cond_2d = torch.randn(3, 2 * dim)

    output = adaln(x, cond_2d)
    assert output.shape == (3, 5, dim)
    assert not torch.isnan(output).any()


def test_when_adaln_forward_backward_then_computes_gradients_for_x_and_cond() -> None:
    dim = 16
    adaln = AdaLN(dim=dim)
    x = torch.randn(2, 4, dim, requires_grad=True)
    cond = torch.randn(2, 4, 2 * dim, requires_grad=True)

    out = adaln(x, cond)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert cond.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(cond.grad).any()


# --- LearnableFourierFeatures Tests ---


def test_when_learnable_fourier_features_initialized_with_odd_f_dim_then_raises_assertion() -> None:
    with pytest.raises(AssertionError, match="must be divisible by 2"):
        LearnableFourierFeatures(pos_dim=4, f_dim=15, h_dim=32, d_dim=64)


def test_when_learnable_fourier_features_initialized_with_incompatible_g_dim_then_raises_assertion() -> None:
    with pytest.raises(AssertionError, match="must be divisible by the number of G dimension"):
        LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64, g_dim=5)


def test_when_learnable_fourier_features_forward_with_include_input_true_then_outputs_expected_shape() -> None:
    lff = LearnableFourierFeatures(
        pos_dim=4, f_dim=16, h_dim=32, d_dim=64, g_dim=2, include_input=True
    )
    pos = torch.randn(2, 5, 2, 4)  # (B, L, G=2, M=4)
    out = lff(pos)
    assert out.shape == (2, 5, 64)
    assert not torch.isnan(out).any()


def test_when_learnable_fourier_features_forward_with_include_input_false_then_outputs_expected_shape() -> None:
    lff = LearnableFourierFeatures(
        pos_dim=4, f_dim=16, h_dim=32, d_dim=64, g_dim=1, include_input=False
    )
    pos = torch.randn(2, 5, 1, 4)
    out = lff(pos)
    assert out.shape == (2, 5, 64)
    assert not torch.isnan(out).any()


def test_when_learnable_fourier_features_backward_then_updates_parameters() -> None:
    lff = LearnableFourierFeatures(pos_dim=4, f_dim=16, h_dim=32, d_dim=64)
    pos = torch.randn(2, 3, 1, 4)

    out = lff(pos)
    loss = out.pow(2).sum()
    loss.backward()

    assert lff.Wr.grad is not None
    for param in lff.mlp.parameters():
        assert param.grad is not None


# --- FourierFeatures Tests ---


def test_when_fourier_features_initialized_with_odd_f_dim_then_raises_assertion() -> None:
    with pytest.raises(AssertionError, match="must be divisible by 2"):
        FourierFeatures(pos_dim=2, f_dim=7)


def test_when_fourier_features_train_false_then_registers_non_trainable_buffer() -> None:
    ff = FourierFeatures(pos_dim=2, f_dim=16, train=False)
    assert hasattr(ff, "B")
    assert not isinstance(ff.B, nn.Parameter)
    assert "B" in dict(ff.named_buffers())


def test_when_fourier_features_train_true_then_registers_trainable_parameter() -> None:
    ff = FourierFeatures(pos_dim=2, f_dim=16, train=True)
    assert isinstance(ff.B, nn.Parameter)
    assert ff.B.requires_grad


def test_when_fourier_features_forward_with_higher_dim_input_then_encodes_correctly() -> None:
    ff = FourierFeatures(pos_dim=3, f_dim=16, include_input=True)
    pos_4d = torch.randn(2, 4, 4, 3)  # (B, H, W, pos_dim)
    enc = ff(pos_4d)
    assert enc.shape == (2, 4, 4, 16 + 3)
    assert not torch.isnan(enc).any()


# --- PositionalEncoding Tests ---


def test_when_positional_encoding_initialized_with_odd_enc_dim_then_raises_assertion() -> None:
    with pytest.raises(AssertionError, match="must be even"):
        PositionalEncoding(pos_dim=1, enc_dim=7)


def test_when_positional_encoding_forward_without_include_input_then_matches_enc_dim() -> None:
    pe = PositionalEncoding(pos_dim=2, enc_dim=16, include_input=False)
    pos = torch.randn(2, 6, 2)
    enc = pe(pos)
    assert enc.shape == (2, 6, 16)
    assert not torch.isnan(enc).any()


def test_when_positional_encoding_forward_with_include_input_then_concatenates_pos() -> None:
    pe = PositionalEncoding(pos_dim=2, enc_dim=16, include_input=True)
    pos = torch.randn(2, 6, 2)
    enc = pe(pos)
    assert enc.shape == (2, 6, 16 + 2)


# --- RewardEncoder Tests ---


def test_when_reward_encoder_same_dim_then_uses_identity_projection() -> None:
    encoder = RewardEncoder(dim=32, hidden_dim=32)
    assert isinstance(encoder.x_proj, nn.Identity)


def test_when_reward_encoder_different_dim_then_uses_linear_projection() -> None:
    encoder = RewardEncoder(dim=32, hidden_dim=64)
    assert isinstance(encoder.x_proj, nn.Linear)
    assert encoder.x_proj.in_features == 32
    assert encoder.x_proj.out_features == 64


def test_when_reward_encoder_forward_with_2d_rewards_then_processes_correctly() -> None:
    encoder = RewardEncoder(dim=32, hidden_dim=32)
    x = torch.randn(2, 5, 32)
    reward_2d = torch.tensor([[0.5, 0.2, 0.9, 0.1, 0.4], [0.3, 0.7, 0.1, 0.8, 0.5]])

    output = encoder(x, reward_2d)
    assert output.shape == (2, 5, 32)
    assert not torch.isnan(output).any()


def test_when_reward_encoder_forward_with_3d_multi_rewards_then_computes_stats_correctly() -> None:
    encoder = RewardEncoder(dim=32, hidden_dim=32)
    x = torch.randn(2, 4, 32)
    # (B=2, L=4, K=3) multi-reward tensor
    reward_3d = torch.randn(2, 4, 3)

    output = encoder(x, reward_3d)
    assert output.shape == (2, 4, 32)
    assert not torch.isnan(output).any()


def test_when_reward_encoder_forward_with_constant_rewards_then_produces_valid_output() -> None:
    encoder = RewardEncoder(dim=16, hidden_dim=16)
    x = torch.randn(2, 3, 16)
    # All reward values equal to 1.0 (median == mean == max == min)
    reward_const = torch.ones(2, 3, 4)

    output = encoder(x, reward_const)
    assert output.shape == (2, 3, 16)
    assert not torch.isnan(output).any()


def test_when_reward_encoder_device_or_dtype_mismatched_then_auto_aligns() -> None:
    encoder = RewardEncoder(dim=16, hidden_dim=16)
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)

    # reward_values in float32
    reward_values = torch.randn(2, 3, 2, dtype=torch.float32)

    encoder.to(dtype=torch.bfloat16)
    output = encoder(x, reward_values)

    assert output.shape == (2, 3, 16)
    assert output.dtype == torch.bfloat16
    assert not torch.isnan(output).any()


def test_when_reward_encoder_backward_called_then_gradients_flow_to_inputs_and_params() -> None:
    encoder = RewardEncoder(dim=16, hidden_dim=16)
    x = torch.randn(2, 3, 16, requires_grad=True)
    reward_values = torch.randn(2, 3, 2, requires_grad=True)

    output = encoder(x, reward_values)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert reward_values.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(reward_values.grad).any()
    for param in encoder.parameters():
        if param.requires_grad:
            assert param.grad is not None


@pytest.mark.parametrize("num_rewards", [1, 2, 5, 20, 100])
def test_when_reward_encoder_receives_unbounded_k_rewards_then_regularizes_and_encodes(
    num_rewards: int,
) -> None:
    encoder = RewardEncoder(dim=32, hidden_dim=64)
    x = torch.randn(2, 8, 32)
    rewards = torch.randn(2, 8, num_rewards)

    out = encoder(x, rewards)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("seq_len", [1, 5, 32, 128])
def test_when_reward_encoder_receives_variable_sequence_lengths_then_executes_successfully(
    seq_len: int,
) -> None:
    encoder = RewardEncoder(dim=32, hidden_dim=64)
    x = torch.randn(2, seq_len, 32)
    rewards = torch.randn(2, seq_len, 4)

    out = encoder(x, rewards)
    assert out.shape == (2, seq_len, 64)
    assert not torch.isnan(out).any()


def test_when_reward_encoder_receives_single_unbatched_embedding_then_modulates_correctly() -> None:
    encoder = RewardEncoder(dim=32, hidden_dim=64)
    # Single 1D embedding (dim,) and 1D rewards (K=5,)
    x_single = torch.randn(32)
    rewards_single = torch.randn(5)

    out = encoder(x_single, rewards_single)
    assert out.shape == (64,)
    assert not torch.isnan(out).any()


def test_when_reward_encoder_receives_single_step_batched_embeddings_then_modulates_correctly() -> None:
    encoder = RewardEncoder(dim=32, hidden_dim=64)
    # Single step batched embeddings (B=4, dim=32) and 2D rewards (B=4, K=8)
    x_batched = torch.randn(4, 32)
    rewards_batched = torch.randn(4, 8)

    out = encoder(x_batched, rewards_batched)
    assert out.shape == (4, 64)
    assert not torch.isnan(out).any()
