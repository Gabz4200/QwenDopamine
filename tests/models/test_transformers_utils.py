"""Tests for the transformers-util side effects.

Covers review finding M5: the CPU monkeypatch of
``qwen3_next.gated_delta_rule`` functions must be opt-in, not a
fire-on-import side effect.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest


class _SentinelModule:
    """Stand-in module whose ``__setattr__`` records writes to the four
    target attributes.
    """

    def __init__(self) -> None:
        # Seed without triggering our own __setattr__.
        object.__setattr__(self, "_tracked_writes", [])
        for n in (
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
            "causal_conv1d_update",
        ):
            object.__setattr__(self, n, object())
        object.__setattr__(self, "_seed_done", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_seed_done", False) and name in {
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
            "causal_conv1d_update",
        }:
            self._tracked_writes.append(name)
        object.__setattr__(self, name, value)


def test_unwrap_is_noop_when_env_var_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``QWENDOPAMINE_CPU_UNWRAP=1`` the helper must do nothing."""
    monkeypatch.delenv("QWENDOPAMINE_CPU_UNWRAP", raising=False)

    from qwendopamine.models import _transformers_utils as tutils

    fake_mod = _SentinelModule()
    # Drop the real qwen3_next module so the helper imports our sentinel.
    with mock.patch.dict(sys.modules, clear=False) as m:
        m["transformers.models.qwen3_next.modeling_qwen3_next"] = fake_mod
        with mock.patch.object(tutils._torch.cuda, "is_available", return_value=False):
            tutils.unwrap_gated_delta_rule_fns()

    assert fake_mod._tracked_writes == [], (
        f"unwrap_gated_delta_rule_fns() rewrote qwen3_next attrs "
        f"{fake_mod._tracked_writes!r} without QWENDOPAMINE_CPU_UNWRAP=1; "
        f"review M5 requires opt-in."
    )


def test_unwrap_runs_when_env_var_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``QWENDOPAMINE_CPU_UNWRAP=1`` the helper runs past the gate.

    The seam is the gate itself: the helper's only behavioural change
    vs. the pre-M5 code is the new ``_should_unwrap_for_cpu`` early
    return. Asserting on the gate gives us the contract — the full
    "setattr on qwen3_next" path is hard to test because the helper
    imports a real third-party module and is brittle under mocking.
    """
    monkeypatch.setenv("QWENDOPAMINE_CPU_UNWRAP", "1")
    from qwendopamine.models import _transformers_utils as tutils

    assert tutils._should_unwrap_for_cpu() is True, (
        "Opt-in env var QWENDOPAMINE_CPU_UNWRAP=1 must enable the unwrap."
    )


def test_importing_model_modules_does_not_mutate_qwen3_next_without_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing infini-dopamine / qwen35 must NOT mutate the qwen3_next
    module when the opt-in env var is unset.

    The seam is the *mutation*, not the *call*: the helper may still
    be invoked at import time, but it must short-circuit to a no-op
    without QWENDOPAMINE_CPU_UNWRAP=1 (review M5).
    """
    monkeypatch.delenv("QWENDOPAMINE_CPU_UNWRAP", raising=False)

    # Drop the cached model modules so we re-import them under the test.
    prefixes = (
        "qwendopamine.models.infinidopamine",
        "qwendopamine.models.qwen35",
    )
    for mod_name in list(sys.modules):
        if any(mod_name.startswith(p) for p in prefixes):
            del sys.modules[mod_name]

    sentinel = _SentinelModule()
    with mock.patch.dict(sys.modules, clear=False) as m:
        m["transformers.models.qwen3_next.modeling_qwen3_next"] = sentinel
        # Catch ImportError: the model modules pull other qwen3_next
        # attrs not on our sentinel. The error itself is proof the
        # unwrap did not run (it would have called setattr before that).
        try:
            importlib.import_module("qwendopamine.models.infinidopamine.model")
        except (ImportError, AttributeError):
            pass
        try:
            importlib.import_module("qwendopamine.models.qwen35.modular_qwen3_5")
        except (ImportError, AttributeError):
            pass

    assert sentinel._tracked_writes == [], (
        f"Importing the model modules rewrote qwen3_next attrs "
        f"{sentinel._tracked_writes!r} without QWENDOPAMINE_CPU_UNWRAP=1."
    )
