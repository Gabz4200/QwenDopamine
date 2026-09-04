"""Behavioral tests for Hugging Face integrations."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from qwendopamine.integrations.gguf import _map_gguf_name_to_hf
from qwendopamine.integrations.huggingface import (
    GDN2HFConfig,
    HFIntegration,
)
from qwendopamine.integrations.tokenizer import load_qwen35_tokenizer
from qwendopamine.models.gdn2.config import GDN2Config


def test_when_make_quantization_config_int8_then_returns_bitsandbytes_config() -> None:
    qconfig = HFIntegration.make_quantization_config(method="int8", device="cpu")
    assert getattr(qconfig, "load_in_8bit", False) is True


def test_when_make_quantization_config_int4_then_returns_4bit_config() -> None:
    qconfig = HFIntegration.make_quantization_config(
        method="int4", compute_dtype="bfloat16"
    )
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
    hf_cfg = HFIntegration.build_gdn2_hf_config(
        hidden_size=64, num_heads=2, head_dim=32
    )
    block = HFIntegration.build_gdn2_hf_block(hf_cfg, layer_idx=0)
    x = torch.randn(2, 4, 64)
    cache = DynamicCache()
    out, attn, past_cache = block(x, past_key_values=cache, use_cache=True)
    assert out.shape == (2, 4, 64)
    assert attn is None
    assert past_cache is not None


def test_when_register_gdn2_hf_called_then_autoconfig_resolves_gdn2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HFIntegration.register_gdn2_hf()
    try:
        cfg = AutoConfig.for_model("gdn2", hidden_size=128, num_heads=4, head_dim=32)
        assert isinstance(cfg, GDN2HFConfig)
        assert cfg.hidden_size == 128
    finally:
        # Teardown: remove gdn2 from AutoConfig registry to avoid side effects
        if hasattr(AutoConfig, "_model_mapping"):
            AutoConfig._model_mapping._extra_content.pop("gdn2", None)


def test_when_map_gguf_name_to_hf_called_then_maps_expected_keys() -> None:
    assert _map_gguf_name_to_hf("token_embd.weight") == "model.embed_tokens.weight"
    assert _map_gguf_name_to_hf("output.weight") == "lm_head.weight"
    assert _map_gguf_name_to_hf("output_norm.weight") == "model.norm.weight"
    assert (
        _map_gguf_name_to_hf("blk.0.attn_q.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        _map_gguf_name_to_hf("blk.3.ssm_conv1d.bias")
        == "model.layers.3.linear_attn.conv1d.bias"
    )
    assert _map_gguf_name_to_hf("unknown.tensor.name") is None


def test_when_composite_multimodal_config_passed_to_causal_lm_then_unwraps_text_config() -> (
    None
):
    r"""Ensure Qwen3_5ForCausalLM and InfiniDopamineForCausalLM unwrap text_config if given a composite config."""
    from types import SimpleNamespace

    from qwendopamine.models.infinidopamine import (
        InfiniDopamineForCausalLM,
        InfiniDopamineTextConfig,
    )
    from qwendopamine.models.qwen35 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    text_cfg = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        intermediate_size_linear=64,
        vocab_size=100,
        layer_types=["linear_attention", "full_attention"],
    )
    composite_cfg = SimpleNamespace(
        model_type="qwen3_5",
        text_config=text_cfg,
    )

    model = Qwen3_5ForCausalLM(composite_cfg)
    assert model.config.vocab_size == 100
    assert model.model.embed_tokens.num_embeddings == 100

    infini_text_cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        intermediate_size_linear=64,
        vocab_size=100,
        layer_types=["linear_attention", "full_attention"],
    )
    composite_infini_cfg = SimpleNamespace(
        model_type="infinidopamine",
        text_config=infini_text_cfg,
    )

    infini_model = InfiniDopamineForCausalLM(composite_infini_cfg)
    assert infini_model.config.vocab_size == 100
    assert infini_model.model.embed_tokens.num_embeddings == 100


def test_when_load_qwen35_tokenizer_all_candidates_fail_then_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoTokenizer

    def _mock_from_pretrained(*args: Any, **kwargs: Any) -> Any:
        raise OSError("Simulated connection/file error")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _mock_from_pretrained)

    with pytest.raises(RuntimeError, match="Failed to load Qwen3.5 tokenizer"):
        load_qwen35_tokenizer("nonexistent/invalid-model-name")


def test_when_register_infinidopamine_hf_called_then_autoconfig_and_automodel_resolve() -> (
    None
):
    from transformers import AutoModelForCausalLM

    from qwendopamine.models.infinidopamine import (
        InfiniDopamineForCausalLM,
        InfiniDopamineTextConfig,
    )

    HFIntegration.register_infinidopamine_hf()

    cfg = AutoConfig.for_model(
        "infinidopamine_text",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        sliding_window=4,
        vocab_size=1000,
    )
    assert isinstance(cfg, InfiniDopamineTextConfig)

    model = AutoModelForCausalLM.from_config(cfg)
    assert isinstance(model, InfiniDopamineForCausalLM)


def test_when_build_infinidopamine_helpers_called_then_instantiates_working_causal_lm() -> (
    None
):
    from qwendopamine.models.infinidopamine import (
        InfiniDopamineForCausalLM,
        InfiniDopamineTextConfig,
    )

    cfg = HFIntegration.build_infinidopamine_config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        sliding_window=4,
        vocab_size=500,
    )
    assert isinstance(cfg, InfiniDopamineTextConfig)

    model = HFIntegration.build_infinidopamine_model(cfg)
    assert isinstance(model, InfiniDopamineForCausalLM)

    inputs = torch.randint(0, 500, (2, 8))
    outputs = model(inputs)
    assert outputs.logits.shape == (2, 8, 500)


def test_when_prepare_model_for_trl_training_called_then_configures_gradient_checkpointing_and_cache() -> (
    None
):
    from qwendopamine.models.infinidopamine import (
        InfiniDopamineForCausalLM,
        InfiniDopamineTextConfig,
    )

    cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        head_dim=16,
        sliding_window=4,
        vocab_size=500,
    )
    model = InfiniDopamineForCausalLM(cfg)

    prepared_model = HFIntegration.prepare_model_for_trl_training(
        model, use_gradient_checkpointing=True
    )
    assert getattr(prepared_model.config, "use_cache", True) is False

    # Verify input embeddings require gradients when checkpointed
    emb = prepared_model.get_input_embeddings()
    assert emb is not None
    emb_out = emb(torch.tensor([[1, 2]]))
    assert emb_out.requires_grad is True
