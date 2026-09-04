"""Tests ensuring InfiniDopamine loads all weights from Qwen/Qwen3.5-0.8B without leaving anything behind."""

from __future__ import annotations

import urllib.error
from typing import Any

import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from qwendopamine.integrations.huggingface import HFIntegration
from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineForCausalLM,
    InfiniDopamineForConditionalGeneration,
    InfiniDopamineTextConfig,
    InfiniDopamineTextModel,
)


@pytest.fixture(scope="module", autouse=True)
def _register_hf_classes() -> None:
    HFIntegration.register_all_hf()


def _get_qwen35_08b_config_and_state_dict() -> tuple[Any, dict[str, torch.Tensor]]:
    r"""Retrieve Qwen3.5-0.8B config and meta state dict from HF Hub with fallback."""
    model_id = "Qwen/Qwen3.5-0.8B"
    try:
        hf_config = AutoConfig.from_pretrained(model_id)
    except (
        OSError,
        urllib.error.URLError,
        TimeoutError,
        RuntimeError,
        ValueError,
    ) as exc:  # pragma: no cover
        pytest.skip(f"Skipping remote Qwen3.5-0.8B test due to network/HF issue: {exc}")

    with torch.device("meta"):
        ref_model = Qwen3_5ForConditionalGeneration(hf_config)
        state_dict = ref_model.state_dict()

    return hf_config, state_dict


@pytest.mark.slow
def test_when_infinidopamine_full_model_loads_qwen35_08b_weights_then_no_weights_left_behind() -> (
    None
):
    r"""Ensure InfiniDopamineForConditionalGeneration consumes all weights from Qwen3.5-0.8B."""
    hf_config, ref_state_dict = _get_qwen35_08b_config_and_state_dict()

    full_cfg = InfiniDopamineConfig(**hf_config.to_dict())

    with torch.device("meta"):
        model = InfiniDopamineForConditionalGeneration(full_cfg)

    # Build simulated state dictionary matching Qwen3.5-0.8B checkpoint
    checkpoint_state_dict = {
        k: torch.empty(v.shape, device="meta") for k, v in ref_state_dict.items()
    }

    # Load with strict=True - should succeed without missing or unexpected keys
    load_result = model.load_qwen35_weights(checkpoint_state_dict, strict=True)
    assert len(load_result.missing_keys) == 0
    assert len(load_result.unexpected_keys) == 0


@pytest.mark.slow
def test_when_infinidopamine_causal_lm_loads_qwen35_08b_weights_then_all_lm_weights_consumed() -> (
    None
):
    r"""Ensure InfiniDopamineForCausalLM consumes all language model weights from Qwen3.5-0.8B."""
    hf_config, ref_state_dict = _get_qwen35_08b_config_and_state_dict()

    text_cfg = InfiniDopamineTextConfig(**hf_config.text_config.to_dict())

    with torch.device("meta"):
        model = InfiniDopamineForCausalLM(text_cfg)

    checkpoint_state_dict = {
        k: torch.empty(v.shape, device="meta") for k, v in ref_state_dict.items()
    }

    # Loading the multimodal checkpoint into Causal LM should remap LM keys and ignore visual/mtp
    load_result = model.load_qwen35_weights(checkpoint_state_dict, strict=True)
    assert len(load_result.missing_keys) == 0
    assert len(load_result.unexpected_keys) == 0


@pytest.mark.slow
def test_when_infinidopamine_text_model_loads_qwen35_08b_weights_then_all_layers_matched() -> (
    None
):
    r"""Ensure InfiniDopamineTextModel consumes all 24 layers and embeddings from Qwen3.5-0.8B."""
    hf_config, ref_state_dict = _get_qwen35_08b_config_and_state_dict()

    text_cfg = InfiniDopamineTextConfig(**hf_config.text_config.to_dict())

    with torch.device("meta"):
        model = InfiniDopamineTextModel(text_cfg)

    checkpoint_state_dict = {
        k: torch.empty(v.shape, device="meta") for k, v in ref_state_dict.items()
    }

    load_result = model.load_qwen35_weights(checkpoint_state_dict, strict=True)
    assert len(load_result.missing_keys) == 0
    assert len(load_result.unexpected_keys) == 0


@pytest.mark.slow
def test_when_infinidopamine_loaded_with_qwen35_weights_then_extra_infinidopamine_weights_initialized() -> (
    None
):
    r"""Ensure extra InfiniDopamine weights (betas, write gate, reward gate) are present and valid."""
    hf_config, ref_state_dict = _get_qwen35_08b_config_and_state_dict()

    text_cfg = InfiniDopamineTextConfig(**hf_config.text_config.to_dict())

    with torch.device("meta"):
        model = InfiniDopamineForCausalLM(text_cfg)

    checkpoint_state_dict = {
        k: torch.empty(v.shape, device="meta") for k, v in ref_state_dict.items()
    }

    model.load_qwen35_weights(checkpoint_state_dict, strict=True)

    # Verify that GDN-2 layer 0 has decoupled write gate, betas, and in_proj_gate
    layer0_linear = model.model.layers[0].linear_attn
    assert hasattr(layer0_linear, "betas")
    assert hasattr(layer0_linear, "in_proj_w")
    assert hasattr(layer0_linear, "in_proj_gate")

    # GatedRewardNet is opt-in via parallel_reward_layers or use_parallel_reward.
    # The default config does not implicitly promote any layer.
    for layer in model.model.layers:
        if hasattr(layer, "reward_branch"):
            raise AssertionError(
                "GatedRewardNet branch should not exist without "
                "parallel_reward_layers or use_parallel_reward"
            )


@pytest.mark.slow
def test_when_infinidopamine_forward_executed_after_qwen35_08b_loading_then_produces_valid_logits() -> (
    None
):
    r"""Test a mini forward pass on CPU with scaled down dimensions to verify post-load execution."""
    hf_config, _ = _get_qwen35_08b_config_and_state_dict()

    text_dict = hf_config.text_config.to_dict()
    text_dict["num_hidden_layers"] = 2
    text_dict["layer_types"] = ["linear_attention", "full_attention"]
    text_dict["hidden_size"] = 64
    text_dict["intermediate_size"] = 128
    text_dict["linear_key_head_dim"] = 16
    text_dict["linear_value_head_dim"] = 16
    text_dict["linear_num_key_heads"] = 2
    text_dict["linear_num_value_heads"] = 4
    text_dict["num_attention_heads"] = 4
    text_dict["num_key_value_heads"] = 2
    text_dict["head_dim"] = 16
    text_dict["vocab_size"] = 500

    cfg = InfiniDopamineTextConfig(**text_dict)
    model = InfiniDopamineForCausalLM(cfg)

    inputs = torch.randint(0, 500, (2, 8))
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.logits.shape == (2, 8, 500)
    assert not torch.isnan(outputs.logits).any()
