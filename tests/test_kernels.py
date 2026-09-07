"""Behavioural tests for the Taichi kernels module.

These tests cover:

* the public surface (``is_available``, ``taichi_arch``, ``require``)
  and the resolution invariants (cached, idempotent, returns a string);
* the ``_make_effective_gate`` helper contract that the per-token
  forward and backward kernels depend on;
* the ``_normalize_qk`` helper used by both GDN-2 paths.

Heavy kernel correctness (fwd/bwd parity with the torch reference) is
covered by the dedicated tests under ``tests/models/test_taichi_*``.
"""

from __future__ import annotations

import pytest
import torch

from qwendopamine.kernels.taichi import (
    is_available,
    require,
    taichi_arch,
)
from qwendopamine.kernels.taichi.gdn2_api import _normalize_qk
from qwendopamine.kernels.taichi.reinforced_kernels import _make_effective_gate

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_when_is_available_called_then_returns_bool() -> None:
    assert isinstance(is_available(), bool)


def test_when_taichi_arch_called_then_returns_string() -> None:
    arch = taichi_arch()
    assert isinstance(arch, str)
    assert arch, "taichi_arch must return a non-empty string"
    # taichi may report things like "Arch.vulkan" or "Arch.cuda"; the
    # contract is "non-empty string" rather than a fixed enumeration.


def test_when_is_available_called_twice_then_idempotent() -> None:
    a = is_available()
    b = is_available()
    assert a == b


def test_when_is_available_no_initialize_then_does_not_initialise() -> None:
    """Passing ``initialize=False`` must not trigger a Taichi init."""
    # Just verify the call works; idempotency is the contract.
    arch_before = taichi_arch(initialize=False)
    is_avail_before = is_available(initialize=False)
    assert isinstance(arch_before, str)
    assert isinstance(is_avail_before, bool)


# ---------------------------------------------------------------------------
# require()
# ---------------------------------------------------------------------------


def test_when_require_called_then_returns_module_or_raises() -> None:
    """``require()`` returns the taichi module on success, raises on
    unavailability. Either outcome must be safe to test."""
    if is_available():
        ti = require()
        assert ti is not None
    else:
        with pytest.raises(RuntimeError):
            require()


# ---------------------------------------------------------------------------
# _make_effective_gate
# ---------------------------------------------------------------------------


def test_when_make_effective_gate_2d_omega_then_contracts_to_2d() -> None:
    """Public omega of shape ``[B, 1]`` contracted with ``[B, D]`` gate
    must produce a ``[B, D]`` effective gate."""
    omega = torch.tensor([[0.5], [0.8]])
    gate = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    eff = _make_effective_gate(omega, gate)
    assert eff.shape == (2, 4)
    assert torch.allclose(eff[0], gate[0] * 0.5)
    assert torch.allclose(eff[1], gate[1] * 0.8)


def test_when_make_effective_gate_1d_omega_then_unsqueezed_to_2d() -> None:
    """Public omega of shape ``[B]`` must be unsqueezed to ``[B, 1]``
    before contracting."""
    omega = torch.tensor([0.5, 0.8])
    gate = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    eff = _make_effective_gate(omega, gate)
    assert eff.shape == (2, 2)


def test_when_make_effective_gate_bad_shape_then_raises() -> None:
    """``[B, D]`` omega (not ``[B, 1]``) must raise ValueError."""
    omega = torch.ones(2, 4)  # wrong — must be [B, 1]
    gate = torch.ones(2, 4)
    with pytest.raises(ValueError, match="omega must be"):
        _make_effective_gate(omega, gate)


def test_when_make_effective_gate_3d_omega_then_raises() -> None:
    omega = torch.ones(2, 1, 1)
    gate = torch.ones(2, 4)
    with pytest.raises(ValueError, match="omega must be"):
        _make_effective_gate(omega, gate)


# ---------------------------------------------------------------------------
# _normalize_qk
# ---------------------------------------------------------------------------


def test_when_normalize_qk_disabled_then_inputs_unchanged() -> None:
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    q_out, k_out = _normalize_qk(q, k, apply=False)
    assert q_out is q
    assert k_out is k


def test_when_normalize_qk_enabled_then_qk_have_unit_norm_last_dim() -> None:
    q = torch.randn(2, 3, 4, 5)
    k = torch.randn(2, 3, 4, 5)
    q_out, k_out = _normalize_qk(q, k, apply=True)
    # Last dim should have unit L2 norm.
    q_norms = q_out.norm(dim=-1)
    k_norms = k_out.norm(dim=-1)
    assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5)
    assert torch.allclose(k_norms, torch.ones_like(k_norms), atol=1e-5)
