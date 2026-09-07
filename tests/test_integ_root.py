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


# ---------------------------------------------------------------------------
# Accelerator kernel arg-migration: tensor_arg_indices must include
# `initial_state` (arg 6) for chunk/recurrent ops (review finding M10).
# Without it, a CPU `initial_state` on a CUDA/XPU/MPS call lands in the
# kernel body on the wrong device.
# ---------------------------------------------------------------------------


def test_when_register_accelerator_kernels_then_initial_state_in_arg_indices() -> None:
    """Every GDN-2 op's tensor-arg spec must include arg 6 (``initial_state``).

    Review finding M10: the production spec lists
    ``[0, 1, 2, 3, 4, 5]`` for the four GDN-2 ops, omitting
    ``initial_state`` (arg 6). The migration helper in ``_register_one``
    only moves args whose index is in the list, so a CPU
    ``initial_state`` on a CUDA/XPU/MPS call lands in the kernel body
    on the wrong device. Delta correctly lists ``[0..6]``.

    The contract: the per-op spec is a module-level constant on
    ``qwendopamine.integrations.pytorch.register`` and every GDN-2 op
    spec includes index 6.
    """
    from qwendopamine.integrations.pytorch import register as reg_module

    # The fix introduces a public per-op spec list.
    spec_lists = reg_module.GDN2_ACCEL_TENSOR_ARG_INDICES
    # The four GDN-2 ops must each list [0..6] (7 args).
    assert len(spec_lists) == 4, (
        f"Expected 4 GDN-2 per-op spec lists (chunk, chunk_with_state, "
        f"recurrent, recurrent_with_state); got {len(spec_lists)}."
    )
    for i, spec in enumerate(spec_lists):
        assert spec == [0, 1, 2, 3, 4, 5, 6], (
            f"GDN-2 spec #{i} must be [0..6] (including initial_state); "
            f"got {spec}. Review M10."
        )


# ---------------------------------------------------------------------------
# GGUF loader: report unexpected keys (review finding M6).
# ---------------------------------------------------------------------------


def test_when_gguf_load_then_unexpected_keys_are_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``load_gguf_weights`` must surface (log + return) unexpected keys.

    Review M6: previously the function discarded the unexpected tuple
    from ``load_state_dict(strict=False)``, so a mis-mapping or extra
    GGUF tensor went unnoticed. The fix returns the set and logs a
    warning for keys not in the allowlist.
    """
    import torch

    from qwendopamine.integrations import gguf as gguf_mod

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

        def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
            return (
                list[str](),
                ["unexpected.key.alpha", "unexpected.key.beta"],
            )

    fake = _FakeModel()
    # Bypass the real GGUF build (no file I/O).
    monkeypatch.setattr(
        gguf_mod,
        "_build_state_dict_from_gguf",
        lambda p: dict[str, torch.Tensor](),  # type: ignore[arg-type,return-value]
    )

    with caplog.at_level("WARNING", logger="qwendopamine.integrations.gguf"):
        returned = gguf_mod.load_gguf_weights(fake, "/dummy.gguf")

    assert returned == {"unexpected.key.alpha", "unexpected.key.beta"}, (
        f"Expected both unexpected keys returned; got {returned!r}"
    )
    assert any("unexpected" in rec.message for rec in caplog.records), (
        f"Expected a warning log about unexpected keys; got {[r.message for r in caplog.records]!r}"
    )


def test_when_gguf_load_then_allowed_unexpected_is_silenced(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Keys in ``allowed_unexpected`` must not trigger a warning."""
    import torch

    from qwendopamine.integrations import gguf as gguf_mod

    class _FakeModel(torch.nn.Module):
        def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
            return (list[str](), ["tied.tie_word_embeddings"])

    fake = _FakeModel()
    monkeypatch.setattr(
        gguf_mod,
        "_build_state_dict_from_gguf",
        lambda p: dict[str, torch.Tensor](),  # type: ignore[arg-type,return-value]
    )

    with caplog.at_level("WARNING", logger="qwendopamine.integrations.gguf"):
        returned = gguf_mod.load_gguf_weights(
            fake, "/dummy.gguf", allowed_unexpected={"tied.tie_word_embeddings"}
        )
    assert returned == {"tied.tie_word_embeddings"}
    assert not any("unexpected" in rec.message for rec in caplog.records), (
        f"Expected no warning for allowed key; got {[r.message for r in caplog.records]!r}"
    )
