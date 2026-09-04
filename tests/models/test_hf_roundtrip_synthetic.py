"""Synthetic HF-checkpoint round-trip test.

A real HF checkpoint round-trip (load ``Qwen/Qwen3.5-0.8B``, run a
forward pass, compare to a reference) exists in
``test_infinidopamine_hf_loading.py`` and ``test_qwen35_hf_08b.py``,
but both are skipped by default because they require a network
connection to the Hub.

This test exercises the same round-trip path — build a fresh
:class:`Qwen3_5ForCausalLM` from a tiny config, snapshot its state
dict, build a *second* model, copy the state dict into it, run a
forward pass, and verify the output is finite and matches the
source model — without depending on a network call. It catches the
same class of bug (state-dict shape mismatch, missing parameter) as
a real-checkpoint test, just on a synthetic random state dict.
"""

from __future__ import annotations

import torch

from qwendopamine.models.qwen35 import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def _tiny_qwen35_text_config() -> Qwen3_5TextConfig:
    """Tiny text-only Qwen3.5 config that runs in a few seconds on CPU."""
    return Qwen3_5TextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        intermediate_size=64,
        vocab_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


def test_when_load_random_state_dict_then_forward_produces_finite_output() -> None:
    """Round-trip: snapshot a model, copy its state dict into a fresh
    model, run a forward pass, verify the output is finite and
    matches the source model output."""
    config = _tiny_qwen35_text_config()
    torch.manual_seed(0)
    src = Qwen3_5ForCausalLM(config).eval()
    dst = Qwen3_5ForCausalLM(config).eval()
    state_dict = {k: v.detach().clone() for k, v in src.state_dict().items()}
    result = dst.load_state_dict(state_dict, strict=True)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0

    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        out_src = src(input_ids).logits
        out_dst = dst(input_ids).logits
    # The two models now have identical weights, so outputs must match.
    assert out_src.shape == out_dst.shape == (1, 4, config.vocab_size)
    assert torch.allclose(out_src, out_dst), (
        "State-dict round-trip changed the model output"
    )
    assert torch.isfinite(out_dst).all(), (
        "Forward pass produced non-finite logits after a state-dict round-trip"
    )


def test_when_load_truncated_state_dict_then_load_state_dict_reports_missing() -> None:
    """Truncating the state dict must surface a missing-key report."""
    config = _tiny_qwen35_text_config()
    model = Qwen3_5ForCausalLM(config).eval()
    full = model.state_dict()
    truncated = {k: v for i, (k, v) in enumerate(full.items()) if i < len(full) - 2}
    result = model.load_state_dict(truncated, strict=False)
    assert len(result.missing_keys) > 0, (
        "Truncated state dict must produce missing-key report"
    )
