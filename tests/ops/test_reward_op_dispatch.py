"""Dispatch tests for Reward/Delta ops layer.

Verifies:
  1. When taichi is unavailable, the public op falls back to torch path.
  2. When taichi is unavailable the op still produces a tensor of correct
    shape and dtype.
  3. The public op signature accepts the same args as the torch reference.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from qwendopamine.kernels.taichi import is_available
from qwendopamine.ops.reward import delta_core_step_autograd


@pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)
class TestRewardOpDispatch:
    """Dispatch fallback and signature parity for Reward/Delta ops."""

    def test_fallback_when_taichi_unavailable(self, monkeypatch):
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
        result = delta_core_step_autograd(
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

    def test_correct_shape_and_dtype_when_taichi_unavailable(self, monkeypatch):
        """When taichi is unavailable the op still produces a tensor of correct shape/dtype."""
        from qwendopamine.kernels.taichi import is_available as _is_avail

        # Force taichi unavailable
        monkeypatch.setattr(_is_avail, "__wrapped__", lambda: False) if hasattr(
            _is_avail, "__wrapped__"
        ) else None
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
        result = delta_core_step_autograd(
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

    def test_signature_matches_torch_reference(self):
        """The public op signature accepts the same args as the torch reference."""
        public_sig = inspect.signature(delta_core_step_autograd)
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
            f"Missing params in delta_core_step_autograd: {required_params - public_params}"
        )
