"""Behavioural tests for the integrations package: GGUF mapping, HF
integration facade, PyTorch custom-op registration, and tokenizer
loading helpers.

This complements ``tests/models/test_integrations.py`` (HF weight
translations) and ``tests/ops/test_*`` (PyTorch custom ops) with a
package-level smoke surface and focused GGUF tests.
"""

from __future__ import annotations

import pytest

from qwendopamine.integrations.gguf import _map_gguf_name_to_hf
from qwendopamine.integrations.pytorch import devices, is_registered

# ---------------------------------------------------------------------------
# GGUF name mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gguf_name,expected",
    [
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output.weight", "lm_head.weight"),
        ("output_norm.weight", "model.norm.weight"),
    ],
)
def test_when_gguf_name_is_top_level_then_mapped_to_hf(gguf_name, expected) -> None:
    assert _map_gguf_name_to_hf(gguf_name) == expected


def test_when_gguf_per_block_attn_then_layer_index_substituted() -> None:
    result = _map_gguf_name_to_hf("blk.3.attn_q.weight")
    assert result == "model.layers.3.self_attn.q_proj.weight"


def test_when_gguf_per_block_ffn_then_layer_index_substituted() -> None:
    result = _map_gguf_name_to_hf("blk.7.ffn_up.weight")
    assert result is not None
    assert "model.layers.7" in result
    assert "mlp.up_proj" in result


def test_when_gguf_unknown_name_then_returns_none() -> None:
    assert _map_gguf_name_to_hf("totally.made.up") is None


def test_when_gguf_block_too_short_then_returns_none() -> None:
    # "blk.x" is not a valid block path (less than 3 dot-segments).
    assert _map_gguf_name_to_hf("blk.x") is None


# ---------------------------------------------------------------------------
# PyTorch custom-op registration
# ---------------------------------------------------------------------------


def test_when_pytorch_custom_ops_registered_then_is_registered_true() -> None:
    """The public surface must report that the custom ops are registered."""
    assert is_registered()


def test_when_devices_module_imported_then_exposes_expected_api() -> None:
    """The devices module must expose the cross-hardware detection helpers."""
    assert hasattr(devices, "detect_available_devices")
    assert hasattr(devices, "default_device")
    assert hasattr(devices, "to_active_device")
    assert hasattr(devices, "supports_device")
    assert hasattr(devices, "reset_cache")
    assert callable(devices.detect_available_devices)


# ---------------------------------------------------------------------------
# HF integration facade
# ---------------------------------------------------------------------------


def test_when_hf_integration_imported_then_class_exposed() -> None:
    from qwendopamine.integrations.huggingface import HFIntegration

    assert HFIntegration is not None
    # All facade methods are static.
    for name in (
        "register_all_hf",
        "register_qwen35_hf",
        "register_infinidopamine_hf",
        "register_gdn2_hf",
        "load_model",
        "load_config",
        "load_tokenizer",
        "make_quantization_config",
        "build_infinidopamine_config",
        "build_infinidopamine_model",
        "save_model",
        "prepare_model_for_trl_training",
    ):
        assert hasattr(HFIntegration, name), f"missing facade method {name}"


def test_when_hf_integration_register_called_then_idempotent() -> None:
    """Registering the InfiniDopamine HF class must be idempotent — a
    second call must not raise."""
    from qwendopamine.integrations.huggingface import HFIntegration

    HFIntegration.register_infinidopamine_hf()
    HFIntegration.register_infinidopamine_hf()
