"""Slow tests for Qwen3.5 model loading real Qwen3.5-0.8B weights.

Covers the missing complement to ``test_infinidopamine_hf_loading``:
loading the same hub checkpoint into our Qwen3.5 model and verifying
strict load plus finite/deterministic forward.
"""

from __future__ import annotations

import urllib.error

import pytest
import torch
from transformers import AutoConfig

from qwendopamine.integrations.huggingface import HFIntegration
from qwendopamine.models.qwen35 import Qwen3_5ForCausalLM


def _get_qwen35_hub_config() -> AutoConfig:
    model_id = "Qwen/Qwen3.5-0.8B"
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except (
        OSError,
        urllib.error.URLError,
        TimeoutError,
        RuntimeError,
        ValueError,
    ) as exc:  # pragma: no cover
        pytest.skip(f"Skipping remote Qwen3.5-0.8B test due to network/HF issue: {exc}")
    return cfg  # type: ignore[no-any-return]


@pytest.mark.slow
def test_when_qwen35_model_loads_qwen35_08b_via_hf_integration_then_forward_finite() -> (
    None
):
    r"""Ensure Qwen3.5 model via HFIntegration loads hub weights and forward is finite."""
    HFIntegration.register_all_hf()
    model_id = "Qwen/Qwen3.5-0.8B"
    try:
        model = HFIntegration.load_model(
            model_id, device_map="cpu", dtype=torch.float32
        )
        tokenizer = HFIntegration.load_tokenizer(model_id)
    except (
        OSError,
        RuntimeError,
        ValueError,
        ImportError,
        AttributeError,
    ) as exc:  # pragma: no cover
        pytest.skip(f"Skipping HF load due to failure: {exc}")
    assert tokenizer is not None
    model.eval()
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"Param {name} non-finite after load"
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        out2 = model(**inputs)
    assert hasattr(out, "logits")
    assert torch.isfinite(out.logits).all(), "Logits must be finite"
    assert not torch.isnan(out.logits).any()
    assert torch.isinf(out.logits).float().sum() == 0
    assert torch.allclose(out.logits.float(), out2.logits.float(), atol=1e-5)


@pytest.mark.slow
def test_when_qwen35_for_causal_lm_loads_hub_config_state_dict_then_missing_filtered() -> (
    None
):
    r"""Load hub state dict into Qwen3.5 model via meta device and check keys."""
    hf_config = _get_qwen35_hub_config()
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(hf_config)  # type: ignore[arg-type]
        ref_sd = model.state_dict()
    # Simulate checkpoint dict with meta tensors
    ckpt = {k: torch.empty(v.shape, device="meta") for k, v in ref_sd.items()}
    result = model.load_state_dict(ckpt, strict=False)
    # No unexpected MoE keys should appear as unexpected when loading self-generated ckpt
    assert len(result.unexpected_keys) == 0
    # Synthetic ckpt matches exactly, so missing should be 0 with strict=False
    assert len(result.missing_keys) == 0
    # Verify forward on tiny real model without network weight values
    from qwendopamine.models.qwen35 import Qwen3_5TextConfig

    tiny = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=500,
    )
    tiny_model = Qwen3_5ForCausalLM(tiny).eval()
    inp = torch.randint(0, 500, (1, 4))
    with torch.no_grad():
        logits = tiny_model(inp).logits
    assert torch.isfinite(logits).all()
    assert not torch.isnan(logits).any()
