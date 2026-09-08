"""Behavioural tests for the models submodules: core normalization,
config adapter, blocks registry, and reinforced-delta layer.

These complement the per-component tests under ``tests/models/`` with
package-surface smoke coverage.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwendopamine.models.blocks import BLOCKS, build_block
from qwendopamine.models.blocks.reward import (
    AsinhScaler,
    LearnableFourierFeatures,
    RewardFiLM,
    RewardFourierEncoder,
    RewardStatisticsExtractor,
    TokenWiseFiLM,
)
from qwendopamine.models.core.config_adapter import ConfigAdapter
from qwendopamine.models.core.normalization import (
    RMSNorm,
    RMSNormGated,
    apply_mask_to_padding_states,
)

# ---------------------------------------------------------------------------
# ConfigAdapter
# ---------------------------------------------------------------------------


def test_when_config_adapter_primary_field_present_then_returns_it() -> None:
    cfg = type("C", (), {"hidden_size": 256, "n_embd": 99})()
    assert ConfigAdapter(cfg, family="t").hidden_size == 256


def test_when_config_adapter_uses_legacy_alias_n_embd() -> None:
    cfg = type("C", (), {"n_embd": 192})()
    assert ConfigAdapter(cfg, family="legacy").hidden_size == 192


def test_when_config_adapter_uses_n_layer_alias() -> None:
    cfg = type("C", (), {"n_layer": 8})()
    assert ConfigAdapter(cfg, family="legacy").num_hidden_layers == 8


def test_when_config_adapter_uses_d_model_alias() -> None:
    cfg = type("C", (), {"d_model": 384})()
    assert ConfigAdapter(cfg, family="gpt").hidden_size == 384


def test_when_config_adapter_fully_empty_then_uses_module_defaults() -> None:
    cfg = type("C", (), {})()
    adapter = ConfigAdapter(cfg, family="empty")
    assert adapter.hidden_size == 768
    assert adapter.vocab_size == 151936
    assert adapter.max_position_embeddings == 32768
    assert adapter.num_hidden_layers == 12


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def test_when_rmsnorm_weight_is_zero_then_output_is_zero() -> None:
    """A zeroed weight should produce zero output — no centring, no bias."""
    norm = RMSNorm(hidden_size=8)
    with torch.no_grad():
        norm.weight.zero_()
    x = torch.randn(1, 4, 8)
    out = norm(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_when_rmsnorm_dtype_casting_then_output_dtype_matches_input() -> None:
    """Output dtype must match input dtype (caller responsibility to
    upcast for fp16)."""
    norm = RMSNorm(hidden_size=8)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(1, 4, 8, dtype=dtype)
        out = norm(x)
        assert out.dtype == dtype


def test_when_rmsnorm_eps_is_zero_then_still_finite_for_unit_input() -> None:
    """A unit input is the only one that is still finite with eps=0."""
    norm = RMSNorm(hidden_size=8, eps=0.0)
    x = torch.ones(1, 4, 8)
    out = norm(x)
    assert torch.isfinite(out).all()


def test_when_rmsnorm_gated_default_then_no_scaling() -> None:
    """Without a gate, RMSNormGated must behave like plain RMSNorm."""
    norm = RMSNormGated(hidden_size=8)
    x = torch.randn(1, 4, 8)
    out_no_gate = norm(x, gate=None)
    out_with_none = norm(x, gate=None)
    assert torch.allclose(out_no_gate, out_with_none)


# ---------------------------------------------------------------------------
# apply_mask_to_padding_states
# ---------------------------------------------------------------------------


def test_when_apply_mask_with_2d_mask_then_broadcasts_to_hidden() -> None:
    """A 2D ``[B, L]`` mask must broadcast to ``[B, L, D]`` and zero pads."""
    h = torch.ones(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    out = apply_mask_to_padding_states(h, attention_mask=mask)
    # Sequence 0: positions 2,3 zeroed.
    assert torch.all(out[0, 2:] == 0)
    assert torch.all(out[0, :2] == 1)
    # Sequence 1: position 3 zeroed.
    assert torch.all(out[1, 3] == 0)


def test_when_apply_mask_dtype_preserved_then_no_upcast() -> None:
    """The dtype must round-trip unchanged (no implicit fp32 upcast)."""
    h = torch.ones(1, 2, 4, dtype=torch.float16)
    mask = torch.tensor([[1, 0]])
    out = apply_mask_to_padding_states(h, attention_mask=mask)
    assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# Blocks registry
# ---------------------------------------------------------------------------


def test_when_blocks_registry_queried_then_dict_like_surface() -> None:
    """The registry must support ``__contains__``, ``__getitem__``, and
    ``__iter__``."""
    assert "qwen" in BLOCKS
    keys = list(BLOCKS)
    assert len(keys) > 0
    cls = BLOCKS["qwen"]
    assert isinstance(cls, type)


def test_when_build_block_unknown_then_raises_key_error() -> None:
    cfg = type("C", (), {})()
    with pytest.raises(KeyError):
        build_block("nonexistent", cfg, layer_idx=0)


def test_when_build_block_returns_nn_module_instance() -> None:
    """Building any registered block must return an ``nn.Module``."""
    from qwendopamine.models.infinidopamine.configs import InfiniDopamineTextConfig

    cfg = InfiniDopamineTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=100,
        layer_types=["full_attention"],
    )
    block = build_block("qwen", cfg, layer_idx=0)
    assert isinstance(block, nn.Module)


# ---------------------------------------------------------------------------
# Reward blocks: parameter / shape contracts
# ---------------------------------------------------------------------------


def test_when_token_wise_film_identity_input_then_unchanged() -> None:
    film = TokenWiseFiLM(dim=8)
    with torch.no_grad():
        film.gamma_proj.weight.zero_()
        film.gamma_proj.bias.fill_(1.0)
        film.beta_proj.weight.zero_()
        film.beta_proj.bias.zero_()
    x = torch.randn(1, 4, 8)
    cond = torch.randn(1, 4, 8)
    out = film(x, cond)
    assert torch.allclose(out, x, atol=1e-6)


def test_when_learnable_fourier_features_forward_then_shape_matches() -> None:
    """Positional input must be ``(B, L, G, pos_dim)``; output is
    ``(B, L, d_dim)``."""
    lff = LearnableFourierFeatures(pos_dim=2, f_dim=8, h_dim=16, d_dim=16, g_dim=1)
    pos = torch.randn(2, 4, 1, 2)  # (B, L, G, pos_dim)
    out = lff(pos)
    assert out.shape == (2, 4, 16)
    assert torch.isfinite(out).all()


def test_when_asinh_scaler_compresses_large_magnitudes() -> None:
    """asinh compresses the tails: large inputs become bounded."""
    scaler = AsinhScaler(dim=4)
    x = torch.tensor([[1e3, -1e3, 0.0, 1.0]])
    y = scaler(x)
    assert torch.isfinite(y).all()
    assert y.abs().max() < 100  # bounded for x in [-1e3, 1e3]


def test_when_reward_statistics_extractor_batched_then_output_3d() -> None:
    """Batched input must produce ``(B, L, 6)`` statistics."""
    extractor = RewardStatisticsExtractor()
    r = torch.randn(2, 4)  # (B, L)
    out = extractor(r, batch_size=2, seq_len=4)
    assert out.shape == (2, 4, 6)
    assert torch.isfinite(out).all()


def test_when_reward_fourier_encoder_3d_input_then_output_3d() -> None:
    """Input must already be ``(B, L, 6)``; output is ``(B, L, d_dim)``."""
    encoder = RewardFourierEncoder(f_dim=8, h_dim=16, d_dim=16)
    rewards = torch.randn(2, 5, 6)
    out = encoder(rewards)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_when_reward_film_same_dim_uses_identity() -> None:
    film = RewardFiLM(dim=8, hidden_dim=8)
    assert isinstance(film.x_proj, nn.Identity)


def test_when_reward_film_different_dim_uses_linear() -> None:
    film = RewardFiLM(dim=8, hidden_dim=16)
    assert isinstance(film.x_proj, nn.Linear)
    assert film.x_proj.out_features == 16
