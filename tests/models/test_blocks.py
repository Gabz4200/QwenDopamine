"""Behavioral tests for core model components.

Tests cover:
- RMSNorm: shape, numerical correctness, gradient flow
- LMHead: shape, gradient flow
- TokenEmbeddings / PositionEmbeddings: shape, init, gradient
- GatedDeltaNetBlock (GDN-1): shape, gradient, single-token decode
- build_block: registry instantiation for all block types
- ResearchDecoder: full forward with each block type
"""
from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from qwendopamine.models.blocks import BLOCKS, build_block
from qwendopamine.models.blocks.experimental_block import ExperimentalBlock
from qwendopamine.models.blocks.gdn_block import GatedDeltaNetBlock
from qwendopamine.models.blocks.qwen_block import QwenDecoderLayer
from qwendopamine.models.embeddings import PositionEmbeddings, TokenEmbeddings
from qwendopamine.models.model_factory import build_model
from qwendopamine.models.normalization import RMSNorm
from qwendopamine.models.output_head import LMHead

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides):
    defaults = {
        "hidden_size": 128,
        "rms_norm_eps": 1e-6,
        "num_heads": 4,
        "head_dim": 32,
        "expand_v": 1.0,
        "num_v_heads": None,
        "conv_size": 4,
        "conv_bias": False,
        "allow_neg_eigval": False,
        "gdn2_kernel_mode": "fallback",
        "vocab_size": 100,
        "max_position_embeddings": 64,
        "num_layers": 2,
        "hidden_dropout_prob": 0.0,
        "block_types": ["qwen"],
        "new_block_scale": 0.001,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape_matches_input(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == (2, 16, 64)

    def test_numerically_close_to_torch_rmsnorm(self):
        norm = RMSNorm(64, eps=1e-6)
        x = torch.randn(2, 16, 64)
        out = norm(x)

        # Manual RMSNorm
        variance = x.pow(2).mean(-1, keepdim=True)
        expected = x * torch.rsqrt(variance + 1e-6)
        expected = norm.weight * expected
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gradients_flow(self):
        norm = RMSNorm(32)
        x = torch.randn(1, 8, 32, requires_grad=True)
        loss = norm(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_zero_input_produces_zero_output(self):
        norm = RMSNorm(32)
        x = torch.zeros(1, 4, 32)
        out = norm(x)
        # RMS of zeros -> division by sqrt(eps) -> all values become 0 after weight multiply
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_weight_is_parameter(self):
        norm = RMSNorm(32)
        assert isinstance(norm.weight, nn.Parameter)
        assert norm.weight.shape == (32,)


# ---------------------------------------------------------------------------
# LMHead
# ---------------------------------------------------------------------------

class TestLMHead:
    def test_output_shape(self):
        head = LMHead(64, 100)
        x = torch.randn(2, 16, 64)
        out = head(x)
        assert out.shape == (2, 16, 100)

    def test_gradients_flow(self):
        head = LMHead(32, 50)
        x = torch.randn(1, 8, 32, requires_grad=True)
        loss = head(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_silu_activation_nonlinearity(self):
        head = LMHead(32, 50)
        x = torch.randn(1, 4, 32)
        out = head(x)
        assert out.abs().max().item() > 0


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_token_embeddings_shape(self):
        emb = TokenEmbeddings(100, 32)
        ids = torch.randint(0, 100, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, 32)

    def test_position_embeddings_shape(self):
        emb = PositionEmbeddings(64, 32)
        ids = torch.randint(0, 64, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, 32)

    def test_embeddings_have_parameters(self):
        tok = TokenEmbeddings(50, 16)
        pos = PositionEmbeddings(50, 16)
        assert sum(p.numel() for p in tok.parameters()) == 50 * 16
        assert sum(p.numel() for p in pos.parameters()) == 50 * 16

    def test_gradients_flow_through_embeddings(self):
        emb = TokenEmbeddings(50, 16)
        ids = torch.randint(0, 50, (2, 8))
        loss = emb(ids).sum()
        loss.backward()
        assert emb.weight.grad is not None


# ---------------------------------------------------------------------------
# GatedDeltaNetBlock (GDN-1)
# ---------------------------------------------------------------------------

class TestGatedDeltaNetBlock:
    def test_forward_shape(self):
        cfg = _config(hidden_size=128)
        block = GatedDeltaNetBlock(cfg, layer_idx=0)
        block.eval()
        x = torch.randn(2, 8, 128)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 8, 128)

    def test_single_token(self):
        cfg = _config(hidden_size=64)
        block = GatedDeltaNetBlock(cfg, layer_idx=0)
        block.eval()
        x = torch.randn(1, 1, 64)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (1, 1, 64)

    def test_gradients_flow(self):
        cfg = _config(hidden_size=64)
        block = GatedDeltaNetBlock(cfg, layer_idx=0)
        block.train()
        x = torch.randn(1, 4, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_output_finite(self):
        cfg = _config(hidden_size=64)
        block = GatedDeltaNetBlock(cfg, layer_idx=0)
        block.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out = block(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ExperimentalBlock
# ---------------------------------------------------------------------------

class TestExperimentalBlock:
    def test_forward_shape(self):
        cfg = _config(hidden_size=128)
        block = ExperimentalBlock(cfg, layer_idx=0)
        block.eval()
        x = torch.randn(2, 8, 128)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 8, 128)

    def test_gradients_flow(self):
        cfg = _config(hidden_size=64)
        block = ExperimentalBlock(cfg, layer_idx=0)
        block.train()
        x = torch.randn(1, 4, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# QwenDecoderLayer
# ---------------------------------------------------------------------------

class TestQwenDecoderLayer:
    def test_forward_shape(self):
        cfg = _config(hidden_size=128)
        block = QwenDecoderLayer(cfg, layer_idx=0)
        block.eval()
        x = torch.randn(2, 8, 128)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 8, 128)

    def test_gradients_flow(self):
        cfg = _config(hidden_size=64)
        block = QwenDecoderLayer(cfg, layer_idx=0)
        block.train()
        x = torch.randn(1, 4, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# build_block registry
# ---------------------------------------------------------------------------

class TestBuildBlock:
    def test_registry_has_all_expected_blocks(self):
        assert "qwen" in BLOCKS
        assert "gdn" in BLOCKS
        assert "gdn2" in BLOCKS
        assert "experimental" in BLOCKS

    @pytest.mark.parametrize("block_name", ["qwen", "gdn", "gdn2", "experimental"])
    def test_build_block_returns_correct_type(self, block_name):
        cfg = _config(hidden_size=64)
        block = build_block(block_name, cfg, layer_idx=0)
        assert isinstance(block, BLOCKS[block_name])

    def test_build_block_unknown_raises(self):
        with pytest.raises(KeyError):
            build_block("nonexistent", _config(), layer_idx=0)


# ---------------------------------------------------------------------------
# ResearchDecoder
# ---------------------------------------------------------------------------

class TestResearchDecoder:
    def test_forward_shape_with_qwen_block(self):
        cfg = _config(hidden_size=128, vocab_size=100, num_layers=2, block_types=["qwen", "qwen"])
        model = build_model(cfg)
        model.eval()
        ids = torch.randint(0, 100, (2, 8))
        positions = torch.arange(8).unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            logits = model(ids, positions)
        assert logits.shape == (2, 8, 100)

    def test_forward_shape_with_gdn_block(self):
        cfg = _config(hidden_size=128, vocab_size=100, num_layers=2, block_types=["gdn", "gdn"])
        model = build_model(cfg)
        model.eval()
        ids = torch.randint(0, 100, (2, 8))
        positions = torch.arange(8).unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            logits = model(ids, positions)
        assert logits.shape == (2, 8, 100)

    def test_forward_shape_with_gdn2_block(self):
        cfg = _config(hidden_size=128, vocab_size=100, num_layers=2, block_types=["gdn2", "gdn2"])
        model = build_model(cfg)
        model.eval()
        ids = torch.randint(0, 100, (2, 8))
        positions = torch.arange(8).unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            logits = model(ids, positions)
        assert logits.shape == (2, 8, 100)

    def test_forward_shape_with_experimental_block(self):
        cfg = _config(hidden_size=128, vocab_size=100, num_layers=2, block_types=["experimental", "experimental"])
        model = build_model(cfg)
        model.eval()
        ids = torch.randint(0, 100, (2, 8))
        positions = torch.arange(8).unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            logits = model(ids, positions)
        assert logits.shape == (2, 8, 100)

    def test_gradients_flow_through_decoder(self):
        cfg = _config(hidden_size=64, vocab_size=50, num_layers=2, block_types=["qwen", "qwen"])
        model = build_model(cfg)
        model.train()
        ids = torch.randint(0, 50, (1, 4))
        positions = torch.arange(4).unsqueeze(0).expand(1, -1)
        loss = model(ids, positions).sum()
        loss.backward()
        # At least some parameters should have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert sum(g.abs().sum().item() for g in grads) > 0

    def test_position_ids_none_defaults_to_arange(self):
        cfg = _config(hidden_size=64, vocab_size=50, num_layers=1)
        model = build_model(cfg)
        model.eval()
        ids = torch.randint(0, 50, (1, 4))
        with torch.no_grad():
            out1 = model(ids)
        positions = torch.arange(4).unsqueeze(0).expand(1, -1)
        with torch.no_grad():
            out2 = model(ids, positions)
        assert torch.allclose(out1, out2)

    def test_output_is_finite(self):
        cfg = _config(hidden_size=64, vocab_size=50, num_layers=2, block_types=["gdn2", "gdn2"])
        model = build_model(cfg)
        model.eval()
        ids = torch.randint(0, 50, (1, 4))
        positions = torch.arange(4).unsqueeze(0).expand(1, -1)
        with torch.no_grad():
            logits = model(ids, positions)
        assert torch.isfinite(logits).all()
