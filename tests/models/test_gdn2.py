"""Behavioral and shape tests for hardware-agnostic GatedDeltaNet2 (GDN-2)."""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from qwendopamine.models.blocks.registry import BLOCKS, build_block
from qwendopamine.models.gdn2.config import GDN2Config
from qwendopamine.models.gdn2.gdn2 import (
    GatedDeltaNet2,
    torch_chunk_gdn2,
    torch_recurrent_gdn2,
)


class DummyConfig:
    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        head_dim: int = 32,
        conv_size: int = 4,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_head = num_heads
        self.head_size = head_dim
        self.conv_size = conv_size
        self.norm_eps = rms_norm_eps
        self.rms_norm_eps = rms_norm_eps
        self.allow_neg_eigval = False
        self.expand_v = 1.0


def test_when_gdn2_initialized_with_int_then_creates_valid_module() -> None:
    layer = GatedDeltaNet2(hidden_size=128, num_heads=4, head_dim=32)
    assert isinstance(layer, torch.nn.Module)
    assert layer.hidden_size == 128


def test_when_gdn2_initialized_with_config_then_creates_valid_module() -> None:
    cfg = DummyConfig(hidden_size=128, num_heads=4, head_dim=32)
    layer = GatedDeltaNet2(cfg, layer_idx=0)
    assert isinstance(layer, torch.nn.Module)
    assert layer.hidden_size == 128


def test_when_gdn2_forward_cpu_then_preserves_shape() -> None:
    layer = GatedDeltaNet2(hidden_size=128, num_heads=4, head_dim=32)
    x = torch.randn(2, 16, 128, dtype=torch.float32)
    out, attn, _cache = layer(x)
    assert out.shape == (2, 16, 128)
    assert attn is None


def test_when_gdn2_backward_then_gradients_flow_to_parameters() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _, _ = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 8, 64)
    for name, param in layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_when_torch_recurrent_gdn2_executed_then_matches_expected_output_shape() -> None:
    b, t, h, d_k, d_v = 2, 8, 4, 16, 16
    q = torch.randn(b, t, h, d_k)
    k = torch.randn(b, t, h, d_k)
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k)
    erase_b = torch.sigmoid(torch.randn(b, t, h, d_k))
    write_w = torch.sigmoid(torch.randn(b, t, h, d_v))

    out_rec, state_rec = torch_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=erase_b,
        w=write_w,
        output_final_state=True,
    )
    out_chk, state_chk = torch_chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=erase_b,
        w=write_w,
        output_final_state=True,
    )

    assert out_rec.shape == (b, t, h, d_v)
    assert state_rec is not None and state_chk is not None
    assert state_rec.shape == (b, h, d_k, d_v)
    assert torch.allclose(out_rec, out_chk, atol=1e-5)
    assert torch.allclose(state_rec, state_chk, atol=1e-5)


def test_when_gdn2_forward_with_dict_cache_then_updates_state() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, layer_idx=0)
    x = torch.randn(1, 1, 64)
    cache: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    out, _, _ = layer(x, past_key_values=cache, use_cache=True)
    assert out.shape == (1, 1, 64)
    assert "recurrent_state" in cache
    assert "conv_state" in cache


def test_when_gdn2_forward_with_dynamic_cache_then_runs() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, layer_idx=0)
    x = torch.randn(1, 1, 64)
    dyn_cache = DynamicCache()

    out, _, past_cache = layer(x, past_key_values=dyn_cache, use_cache=True)
    assert out.shape == (1, 1, 64)
    assert past_cache is not None


def test_when_gdn2_registered_in_blocks_then_build_block_instantiates_gdn2() -> None:
    assert "gdn2" in BLOCKS
    assert "qwen35_gdn2" in BLOCKS
    cfg = DummyConfig(hidden_size=64, num_heads=2, head_dim=32)
    module = build_block("gdn2", cfg, layer_idx=0)
    assert isinstance(module, GatedDeltaNet2)


def test_when_gdn2_config_from_name_valid_then_returns_config() -> None:
    cfg = GDN2Config.from_name("gdn2_1.3B")
    assert cfg.hidden_size == 2304


def test_when_gdn2_config_from_name_invalid_then_raises_key_error() -> None:
    with pytest.raises(KeyError, match="Unknown config name"):
        GDN2Config.from_name("invalid_model_name")
