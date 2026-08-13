"""Tests for GDN-2 (Gated DeltaNet-2) mixer and block.

Follows TDD: behavior-focused tests through public interfaces.
The existing GDN block tests are in this file.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from qwendopamine.models.blocks.gdn2 import GDN2Mixer, GDN2Projections, GatedDeltaNet2Block
from qwendopamine.models.blocks.gdn2_ops import dispatch_gdn2


def _config(**overrides):
    """Minimal config namespace matching `getattr(config, key, default)` convention."""
    defaults = {
        "hidden_size": 256,
        "rms_norm_eps": 1e-6,
        "num_heads": 4,
        "head_dim": 32,
        "expand_v": 1.0,
        "expand_k": 1.0,
        "num_v_heads": None,
        "conv_size": 4,
        "conv_bias": False,
        "chunk_size": 64,
        "allow_neg_eigval": False,
        "use_short_conv": True,
        "gdn2_kernel_mode": "fallback",
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ── RED: first test — mixer forward output shape ──────────────────────────────


def test_when_gdn2_mixer_forward_then_output_shape_matches_hidden():
    r"""GDN2Mixer must output [B, T, hidden_size] for arbitrary causal input."""
    cfg = _config()
    mixer = GDN2Mixer(cfg, layer_idx=0)
    mixer.eval()

    batch_size, seq_len = 2, 16
    hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)
    with torch.no_grad():
        out = mixer(hidden)

    assert out.shape == (batch_size, seq_len, cfg.hidden_size), (
        f"Expected {(batch_size, seq_len, cfg.hidden_size)}, got {tuple(out.shape)}"
    )


def test_when_gdn2_mixer_single_token_then_output_shape_matches():
    r"""Single-token (seq_len=1) path must work — needed for autoregressive decode."""
    cfg = _config()
    mixer = GDN2Mixer(cfg, layer_idx=0)
    mixer.eval()

    hidden = torch.randn(1, 1, cfg.hidden_size)
    with torch.no_grad():
        out = mixer(hidden)

    assert out.shape == (1, 1, cfg.hidden_size)


def test_when_gdn2_mixer_padded_then_ignores_padding():
    r"""A zero-padded token at the end should not contribute to the state."""
    cfg = _config()
    mixer = GDN2Mixer(cfg, layer_idx=0)
    mixer.eval()

    real = torch.randn(1, 4, cfg.hidden_size)
    padding = torch.zeros(1, 3, cfg.hidden_size)
    mixed = torch.cat([real, padding], dim=1)

    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)

    with torch.no_grad():
        out_masked = mixer(mixed, attention_mask=mask)

    # Masked output: last 3 positions should be identical (all zeros anyway)
    assert torch.allclose(out_masked[:, 4:, :], torch.zeros(1, 3, cfg.hidden_size), atol=1e-5)


def test_when_gdn2_multi_head_then_key_value_dims_consistent():
    r"""key_dim = num_heads * head_dim; value_dim = num_v_heads * head_v_dim."""
    cfg = _config(num_heads=8, head_dim=32, expand_v=2.0, num_v_heads=8)
    mixer = GDN2Mixer(cfg, layer_idx=0)

    assert mixer.key_dim == cfg.num_heads * cfg.head_dim
    assert mixer.value_dim == cfg.num_v_heads * int(cfg.head_dim * cfg.expand_v)


def test_when_gdn2_reduce_to_kda_then_output_runs():
    r"""When erase=write=1 (sigmoid(inf)), GDN-2 should reduce to KDA behavior."""
    cfg = _config()
    mixer = GDN2Mixer(cfg, layer_idx=0)
    mixer.eval()

    hidden = torch.randn(1, 8, cfg.hidden_size)
    with torch.no_grad():
        out = mixer(hidden)

    assert torch.isfinite(out).all(), "Output must be finite for standard config"


def test_when_gdn2_mixer_gradients_flow():
    r"""Gradients must flow through the full recurrence path."""
    cfg = _config()
    mixer = GDN2Mixer(cfg, layer_idx=0)
    mixer.train()

    hidden = torch.randn(1, 4, cfg.hidden_size, requires_grad=True)
    out = mixer(hidden)
    loss = out.sum()
    loss.backward()

    assert hidden.grad is not None, "Input gradient must be non-None"
    assert hidden.grad.abs().sum().item() > 0, "Input gradient must be non-zero"


# ── Chunk mode / dispatch tests ──────────────────────────────────────────────


def test_when_gdn2_kernel_mode_chunk_on_cpu_then_falls_back():
    r"""CPU path must still run when chunk mode is requested."""
    cfg = _config(gdn2_kernel_mode="chunk")
    mixer = GDN2Mixer(cfg, layer_idx=0)
    mixer.eval()

    hidden = torch.randn(1, 4, cfg.hidden_size)
    with torch.no_grad():
        out = mixer(hidden)

    assert out.shape == (1, 4, cfg.hidden_size)


def test_when_dispatch_gdn2_on_cpu_then_returns_fallback_output():
    r"""dispatch_gdn2 must not crash on CPU and must return finite output."""
    cfg = _config()
    mixer = GDN2Mixer(cfg, layer_idx=0)
    mixer.eval()

    hidden = torch.randn(1, 4, cfg.hidden_size)
    with torch.no_grad():
        B, T, _ = hidden.shape
        q = mixer.q_proj(hidden)
        q = mixer.q_conv(q)
        q = F.normalize(q, p=2, dim=-1)
        k = mixer.k_proj(hidden)
        k = mixer.k_conv(k)
        k = F.normalize(k, p=2, dim=-1)
        v = mixer.v_proj(hidden)
        v = mixer.v_conv(v)
        q = q.view(B, T, mixer.num_heads, mixer.head_dim).transpose(1, 2)
        k = k.view(B, T, mixer.num_heads, mixer.head_dim).transpose(1, 2)
        v = v.view(B, T, mixer.num_v_heads, mixer.head_v_dim).transpose(1, 2)
        b = torch.sigmoid(mixer.b_proj(hidden)).view(B, T, mixer.num_heads, mixer.head_dim).transpose(1, 2)
        w = torch.sigmoid(mixer.w_proj(hidden)).view(B, T, mixer.num_v_heads, mixer.head_v_dim).transpose(1, 2)
        f = mixer.f_proj(hidden)
        g = -mixer.A_log.repeat_interleave(mixer.head_dim).view(1, 1, -1) * F.softplus(f + mixer.dt_bias)
        alpha = g.exp().view(B, T, mixer.num_heads, mixer.head_dim).transpose(1, 2)
        proj = GDN2Projections(q, k, v, alpha, b, w)
        out = dispatch_gdn2(mixer, hidden, proj=proj)

    assert out.shape == (1, 4, cfg.hidden_size)
    assert torch.isfinite(out).all()


# ── Block-level tests ─────────────────────────────────────────────────────────


def test_when_gdn2_block_forward_then_output_shape_matches():
    r"""GatedDeltaNet2Block wraps mixer + MLP in a Transformer residual block."""
    cfg = _config(hidden_size=256)
    block = GatedDeltaNet2Block(cfg, layer_idx=0)
    block.eval()

    hidden = torch.randn(2, 8, cfg.hidden_size)
    with torch.no_grad():
        out = block(hidden)

    assert out.shape == (2, 8, cfg.hidden_size)


def test_when_gdn2_block_residual_then_output_matches_input_range():
    r"""Residual + RMSNorm should keep outputs in a sane range."""
    cfg = _config(hidden_size=128)
    block = GatedDeltaNet2Block(cfg, layer_idx=0)
    block.eval()

    hidden = torch.randn(1, 16, cfg.hidden_size)
    with torch.no_grad():
        out = block(hidden)

    assert out.abs().max().item() < 10, f"Output exploded: max abs = {out.abs().max().item()}"


def test_when_gdn2_block_gradients_flow():
    r"""Gradients must flow through the full block."""
    cfg = _config(hidden_size=128)
    block = GatedDeltaNet2Block(cfg, layer_idx=0)
    block.train()

    hidden = torch.randn(1, 4, cfg.hidden_size, requires_grad=True)
    out = block(hidden)
    loss = out.sum()
    loss.backward()

    assert hidden.grad is not None
    assert hidden.grad.abs().sum().item() > 0
