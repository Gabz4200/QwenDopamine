"""Dispatch tests for Reward/Delta ops layer.

Verifies:
  1. When taichi is unavailable, the public op falls back to torch path.
  2. When taichi is unavailable the op still produces a tensor of correct
    shape and dtype.
  3. The public op signature accepts the same args as the torch reference.
  4. The registered ``delta_core_step`` custom op passes ``opcheck``
    over a battery of input shapes (mirrors the official Python
    custom-ops tutorial).
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch.library import opcheck

from qwendopamine.kernels.taichi import is_available
from qwendopamine.ops.reward import delta_core_step_out


# ---------------------------------------------------------------------------
# Module-level test functions (no class wrapper — matches the rest of
# the test suite layout).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)
def test_fallback_when_taichi_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When taichi is forced unavailable, ops fall back to torch path."""
    # Monkeypatch is_available to return False
    monkeypatch.setattr("qwendopamine.kernels.taichi.is_available", lambda: False)

    B, D = 2, 8
    torch.manual_seed(0)
    state = torch.zeros(B, D, D)
    k = torch.randn(B, D)  # per-token: [B, D]
    v = torch.randn(B, D)  # per-token: [B, D]
    omega_w = torch.full((B, 1), 0.7)  # [B, 1] or [B]
    omega_e = torch.full((B, 1), 0.5)  # [B, 1] or [B]
    write = torch.rand(B, D)  # [B, D]
    erase = torch.rand(B, D)  # [B, D]
    next_state = torch.empty(B, D, D)

    # This should fall back to torch without error
    # The fallback computes the state update via torch ops
    result = delta_core_step_out(
        state,
        k,
        v,
        omega_w,
        omega_e,
        write,
        erase,
        next_state,
    )

    # Verify output shape matches input state shape
    assert result.shape == state.shape, (
        f"Expected result shape {state.shape}, got {result.shape}"
    )

    # Verify dtype is float32
    assert result.dtype == torch.float32, (
        f"Expected result dtype float32, got {result.dtype}"
    )


@pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)
def test_correct_shape_and_dtype_when_taichi_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When taichi is unavailable the op still produces a tensor of correct shape/dtype."""
    monkeypatch.setattr("qwendopamine.kernels.taichi.is_available", lambda: False)

    B, D = 2, 8
    torch.manual_seed(42)
    state = torch.zeros(B, D, D)
    k = torch.randn(B, D)  # per-token: [B, D]
    v = torch.randn(B, D)  # per-token: [B, D]
    omega_w = torch.full((B, 1), 0.7)  # [B, 1] or [B]
    omega_e = torch.full((B, 1), 0.5)  # [B, 1] or [B]
    write = torch.rand(B, D)  # [B, D]
    erase = torch.rand(B, D)  # [B, D]
    next_state = torch.empty(B, D, D)

    # Call the op; it should fall back to torch and produce correct results
    result = delta_core_step_out(
        state,
        k,
        v,
        omega_w,
        omega_e,
        write,
        erase,
        next_state,
    )

    # Verify shape: [B, D, D]
    assert result.shape == torch.Size([B, D, D]), (
        f"Expected result shape [B,D,D]={torch.Size([B, D, D])}, got {result.shape}"
    )

    # Verify dtype: should be float32
    assert result.dtype == torch.float32, (
        f"Expected result dtype float32, got {result.dtype}"
    )


def test_signature_matches_torch_reference() -> None:
    """The public op signature accepts the same args as the torch reference."""
    public_sig = inspect.signature(delta_core_step_out)
    public_params = set(public_sig.parameters.keys())

    # The public op should accept: state, k, v, omega_w, omega_e, write, erase, next_state
    required_params = {
        "state",
        "k",
        "v",
        "omega_w",
        "omega_e",
        "write",
        "erase",
        "next_state",
    }
    assert required_params.issubset(public_params), (
        f"Missing params in delta_core_step_out: {required_params - public_params}"
    )


def test_torch_fallback_matches_pure_torch_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Taichi is forced unavailable, the public op must produce the
    same output as the pure-PyTorch reference helper.

    This proves the fallback path is not just safe (no exception, right
    shape/dtype) but also numerically equivalent to the hand-derived
    reference, so the fallback can be trusted in production.
    """
    monkeypatch.setattr("qwendopamine.kernels.taichi.is_available", lambda: False)

    from qwendopamine.ops.reward import _reward_torch_step

    B, D = 3, 5
    torch.manual_seed(7)
    state = torch.randn(B, D, D)
    k = torch.randn(B, D)
    v = torch.randn(B, D)
    omega_w = torch.full((B, 1), 0.6)
    omega_e = torch.full((B, 1), 0.4)
    write = torch.rand(B, D)
    erase = torch.rand(B, D)
    next_state = torch.empty(B, D, D)

    expected = _reward_torch_step(state, k, v, omega_w, omega_e, write, erase)
    result = delta_core_step_out(
        state, k, v, omega_w, omega_e, write, erase, next_state
    )
    assert torch.allclose(result, next_state), (
        "delta_core_step_out must write into next_state on the torch path"
    )
    assert torch.allclose(result, expected, atol=1e-6), (
        "Torch fallback must match the pure-PyTorch reference"
    )


@pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)
def test_taichi_path_runs_in_no_grad_mode() -> None:
    """Smoke test that the Taichi-backed reward op runs in no_grad mode.

    This is the regression for review 5.5: the production inference
    path runs under ``torch.no_grad()`` and the Taichi kernel must
    not raise. The torch-fallback path is exercised by the test above.
    """
    from qwendopamine.kernels.taichi import delta_core_step_out as taichi_out

    B, D = 2, 4
    torch.manual_seed(11)
    state = torch.zeros(B, D, D)
    k = torch.randn(B, D)
    v = torch.randn(B, D)
    omega_w = torch.full((B, 1), 0.5)
    omega_e = torch.full((B, 1), 0.5)
    write = torch.rand(B, D)
    erase = torch.rand(B, D)
    next_state = torch.empty(B, D, D)
    with torch.no_grad():
        out = taichi_out(state, k, v, omega_w, omega_e, write, erase, next_state)
    assert out.shape == state.shape


# ---------------------------------------------------------------------------
# opcheck battery for delta_core_step (mirrors the official Python
# custom-ops tutorial)
# ---------------------------------------------------------------------------
def _delta_inputs(B: int, D: int) -> tuple[torch.Tensor, ...]:
    """Build a fresh input tuple for the delta_core_step op."""
    torch.manual_seed(0)
    state = torch.zeros(B, D, D)
    k = torch.randn(B, D)
    v = torch.randn(B, D)
    omega_w = torch.zeros(B, 1)
    omega_e = torch.zeros(B, 1)
    write = torch.rand(B, D)
    erase = torch.rand(B, D)
    return (state, k, v, omega_w, omega_e, write, erase)


@pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)
def test_delta_core_step_opcheck_battery() -> None:
    """The registered ``delta_core_step`` op passes ``opcheck`` on a battery."""
    op = torch.ops.qwendopamine.delta_core_step
    base = _delta_inputs(2, 4)
    examples: list[tuple[torch.Tensor, ...]] = [
        base,
        _delta_inputs(1, 0),  # empty D
        tuple(
            t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
            for t in base
        ),  # fp64
    ]
    for example in examples:
        opcheck(op, example, {})
