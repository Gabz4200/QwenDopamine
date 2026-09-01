"""Behavioral tests for normalization layers and masking utilities."""

from __future__ import annotations

import torch

from qwendopamine.models.core.normalization import (
    RMSNorm,
    RMSNormGated,
    apply_mask_to_padding_states,
)


def test_when_rmsnorm_forward_then_scales_variance_to_unit_norm() -> None:
    hidden_size = 64
    norm = RMSNorm(hidden_size=hidden_size, eps=1e-6)
    inputs = torch.randn(2, 8, hidden_size) * 10.0

    output = norm(inputs)

    assert output.shape == inputs.shape
    assert not torch.isnan(output).any()
    # RMSNorm output variance over last dimension should be close to 1.0 (with weight=1.0)
    rms = torch.sqrt(output.pow(2).mean(-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_when_rmsnorm_forward_with_large_fp16_inputs_then_does_not_overflow() -> None:
    r"""Verify RMSNorm calculates variance in fp32 to prevent float16 overflow on large magnitudes."""
    hidden_size = 64
    norm = RMSNorm(hidden_size=hidden_size, eps=1e-6)
    # 500.0^2 = 250,000 which overflows float16 max (65,504) if squared directly in float16
    inputs_fp16 = torch.full((2, 4, hidden_size), 500.0, dtype=torch.float16)

    output = norm(inputs_fp16)

    assert output.dtype == torch.float16
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert torch.allclose(output, torch.ones_like(output, dtype=torch.float16), atol=1e-2)


def test_when_rmsnorm_gated_with_gate_then_applies_silu_gating() -> None:
    hidden_size = 32
    norm_gated = RMSNormGated(hidden_size=hidden_size, eps=1e-6)
    inputs = torch.randn(2, 4, hidden_size)
    gate = torch.randn(2, 4, hidden_size)

    output_gated = norm_gated(inputs, gate=gate)
    output_ungated = norm_gated(inputs, gate=None)

    assert output_gated.shape == inputs.shape
    assert not torch.isnan(output_gated).any()
    # Output with gate should differ from ungated output
    assert not torch.allclose(output_gated, output_ungated)


def test_when_apply_mask_to_padding_states_with_mask_then_zeros_padded_tokens() -> None:
    batch_size, seq_len, hidden_size = 2, 4, 16
    hidden_states = torch.ones(batch_size, seq_len, hidden_size)
    # Mask where second sequence has 2 active tokens and 2 padded tokens
    attention_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])

    masked_states = apply_mask_to_padding_states(
        hidden_states, attention_mask=attention_mask
    )

    assert masked_states.shape == hidden_states.shape
    # Padded positions in second sequence should be zeroed
    assert torch.all(masked_states[1, 2:] == 0.0)
    # Active positions should remain unchanged
    assert torch.all(masked_states[1, :2] == 1.0)


def test_when_apply_mask_to_padding_states_without_mask_then_returns_unchanged() -> (
    None
):
    hidden_states = torch.randn(2, 4, 16)
    result = apply_mask_to_padding_states(hidden_states, attention_mask=None)
    assert torch.equal(result, hidden_states)
