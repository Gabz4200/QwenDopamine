"""Tests for GDN-2 (Gated DeltaNet-2) mixer and block.

Behavior-focused tests through public interfaces.
"""
from __future__ import annotations

import types

import torch

from qwendopamine.models.blocks.gdn2 import GDN2Mixer, GDN2Projections, GatedDeltaNet2Block
from qwendopamine.models.blocks.gdn2_ops import dispatch_gdn2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def make_mixer(**kw):
    cfg = _config(**kw)
    m = GDN2Mixer(cfg, layer_idx=0)
    m.eval()
    return m


def make_block(**kw):
    cfg = _config(**kw)
    b = GatedDeltaNet2Block(cfg, layer_idx=0)
    b.eval()
    return b


# ---------------------------------------------------------------------------
# Shape / dtype
# ---------------------------------------------------------------------------

def test_forward_output_shape_matches_hidden():
    hidden = torch.randn(2, 16, 256)
    with torch.no_grad():
        out = make_mixer()(hidden)
    assert out.shape == (2, 16, 256)


def test_single_token_decode():
    hidden = torch.randn(1, 1, 256)
    with torch.no_grad():
        out = make_mixer()(hidden)
    assert out.shape == (1, 1, 256)


def test_multi_head_dims_consistent():
    m = make_mixer(num_heads=8, head_dim=32, expand_v=2.0, num_v_heads=8)
    assert m.key_dim == 8 * 32
    assert m.value_dim == 8 * int(32 * 2.0)


def test_output_is_finite():
    hidden = torch.randn(1, 8, 256)
    with torch.no_grad():
        out = make_mixer()(hidden)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_mixer_gradients_flow():
    m = make_mixer()
    m.train()
    hidden = torch.randn(1, 4, 256, requires_grad=True)
    m(hidden).sum().backward()
    assert hidden.grad is not None
    assert hidden.grad.abs().sum().item() > 0


def test_block_gradients_flow():
    b = make_block(hidden_size=128)
    b.train()
    hidden = torch.randn(1, 4, 128, requires_grad=True)
    b(hidden).sum().backward()
    assert hidden.grad is not None
    assert hidden.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Block-level shape tests
# ---------------------------------------------------------------------------

def test_block_output_shape():
    hidden = torch.randn(2, 8, 256)
    with torch.no_grad():
        out = make_block(hidden_size=256)(hidden)
    assert out.shape == (2, 8, 256)


def test_block_output_range():
    hidden = torch.randn(1, 16, 128)
    with torch.no_grad():
        out = make_block(hidden_size=128)(hidden)
    assert out.abs().max().item() < 10


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------

def test_chunk_mode_on_cpu_runs():
    hidden = torch.randn(1, 4, 256)
    with torch.no_grad():
        out = make_mixer(gdn2_kernel_mode="chunk")(hidden)
    assert out.shape == (1, 4, 256)


def test_dispatch_gdn2_on_cpu():
    m = make_mixer()
    hidden = torch.randn(1, 4, 256)
    with torch.no_grad():
        B, T, _ = hidden.shape
        q_flat = m._proj_q(hidden)
        k_flat = m._proj_k(hidden)
        v_flat = m._proj_v(hidden)
        b_flat = m._proj_erase_gate(hidden)
        w_flat = m._proj_write_gate(hidden)
        alpha_flat = m._proj_log_decay(hidden)
        q = q_flat.view(B, T, m.num_heads, m.head_dim).transpose(1, 2)
        k = k_flat.view(B, T, m.num_heads, m.head_dim).transpose(1, 2)
        v = v_flat.view(B, T, m.num_v_heads, m.head_v_dim).transpose(1, 2)
        b = b_flat.view(B, T, m.num_heads, m.head_dim).transpose(1, 2)
        w = w_flat.view(B, T, m.num_v_heads, m.head_v_dim).transpose(1, 2)
        alpha = alpha_flat.view(B, T, m.num_heads, m.head_dim).transpose(1, 2)
        proj = GDN2Projections(q, k, v, alpha, b, w)
        out = dispatch_gdn2(m, hidden, proj)

    assert out.shape == (1, 4, 256)
    assert torch.isfinite(out).all()
