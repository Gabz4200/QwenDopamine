"""Behavioral tests for Hugging Face integrations."""

from __future__ import annotations

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from qwendopamine.integrations.huggingface import (
    GDN2HFConfig,
    HFIntegration,
)
from qwendopamine.models.gdn2.config import GDN2Config


def test_when_make_quantization_config_int8_then_returns_bitsandbytes_config() -> None:
    qconfig = HFIntegration.make_quantization_config(method="int8", device="cpu")
    assert getattr(qconfig, "load_in_8bit", False) is True


def test_when_make_quantization_config_int4_then_returns_4bit_config() -> None:
    qconfig = HFIntegration.make_quantization_config(method="int4", compute_dtype="bfloat16")
    assert getattr(qconfig, "load_in_4bit", False) is True
    assert getattr(qconfig, "bnb_4bit_compute_dtype", None) == torch.bfloat16


def test_when_gdn2_hf_config_created_then_roundtrips_to_gdn2_config() -> None:
    gdn_cfg = GDN2Config(hidden_size=128, num_heads=4, head_dim=32)
    hf_cfg = GDN2HFConfig.from_gdn2_config(gdn_cfg)
    assert hf_cfg.hidden_size == 128
    assert hf_cfg.num_heads == 4
    restored_cfg = hf_cfg.to_gdn2_config()
    assert isinstance(restored_cfg, GDN2Config)
    assert restored_cfg.hidden_size == 128


def test_when_gdn2_hf_block_forward_executed_then_preserves_shape_and_cache() -> None:
    hf_cfg = HFIntegration.build_gdn2_hf_config(hidden_size=64, num_heads=2, head_dim=32)
    block = HFIntegration.build_gdn2_hf_block(hf_cfg, layer_idx=0)
    x = torch.randn(2, 4, 64)
    cache = DynamicCache()
    out, attn, past_cache = block(x, past_key_values=cache, use_cache=True)
    assert out.shape == (2, 4, 64)
    assert attn is None
    assert past_cache is not None


def test_when_register_gdn2_hf_called_then_autoconfig_resolves_gdn2() -> None:
    HFIntegration.register_gdn2_hf()
    cfg = AutoConfig.for_model("gdn2", hidden_size=128, num_heads=4, head_dim=32)
    assert isinstance(cfg, GDN2HFConfig)
    assert cfg.hidden_size == 128
