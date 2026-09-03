"""Dispatch tests for GDN-2 ops layer.

Verifies:
  1. When taichi is unavailable, the public op falls back to torch path.
  2. When taichi is unavailable the op still produces a tensor of correct
    shape and dtype.
  3. The public op signature accepts the same args as the torch reference.
  4. The registered custom ops pass ``opcheck`` over a battery of
    representative input shapes (mirrors the official PyTorch tutorial's
    example battery at
    https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html
    "Testing Python custom operators").
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch.library import opcheck

from qwendopamine.kernels.taichi import is_available
from qwendopamine.ops.gdn2 import chunk_taichi_gdn2, recurrent_taichi_gdn2


@pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)
class TestGdn2OpDispatch:
    """Dispatch fallback and signature parity for GDN-2 ops."""

    def test_fallback_when_taichi_unavailable(self, monkeypatch):
        """When taichi is forced unavailable, ops fall back to torch path."""
        # Monkeypatch is_available to return False
        monkeypatch.setattr("qwendopamine.kernels.taichi.is_available", lambda: False)

        B, T, H, K, V = 2, 8, 2, 4, 4
        torch.manual_seed(0)
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        g = torch.randn(B, T, H, K) * -0.1
        b = torch.rand(B, T, H, K)
        w = torch.rand(B, T, H, V)
        init = torch.zeros(B, H, K, V)

        # These should fall back to torch without error
        out_torch, state_torch = recurrent_taichi_gdn2(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=init,
            output_final_state=True,
        )
        out_fallback, state_fallback = chunk_taichi_gdn2(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=init,
            output_final_state=True,
        )

        # Both should produce same-shaped outputs
        assert out_torch.shape == out_fallback.shape, (
            f"Output shape mismatch: torch={out_torch.shape}, fallback={out_fallback.shape}"
        )
        assert state_torch is not None
        assert state_fallback is not None
        assert state_torch.shape == state_fallback.shape, (
            f"State shape mismatch: torch={state_torch.shape}, fallback={state_fallback.shape}"
        )

        # Both should have same dtype
        assert out_torch.dtype == out_fallback.dtype, (
            f"Output dtype mismatch: torch={out_torch.dtype}, fallback={out_fallback.dtype}"
        )
        assert state_torch.dtype == state_fallback.dtype, (
            f"State dtype mismatch: torch={state_torch.dtype}, fallback={state_fallback.dtype}"
        )

    def test_correct_shape_and_dtype_when_taichi_unavailable(self, monkeypatch):
        """When taichi is unavailable the op still produces a tensor of correct shape/dtype."""
        from qwendopamine.kernels.taichi import is_available as _is_avail

        # Force taichi unavailable
        monkeypatch.setattr(_is_avail, "__wrapped__", lambda: False) if hasattr(
            _is_avail, "__wrapped__"
        ) else None
        monkeypatch.setattr("qwendopamine.kernels.taichi.is_available", lambda: False)

        B, T, H, K, V = 2, 8, 2, 4, 4
        torch.manual_seed(42)
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        g = torch.randn(B, T, H, K) * -0.1
        b = torch.rand(B, T, H, K)
        w = torch.rand(B, T, H, V)
        init = torch.zeros(B, H, K, V)

        # Call the ops; they should fall back to torch and produce correct results
        out, state = recurrent_taichi_gdn2(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=init,
            output_final_state=True,
        )

        # Verify shape: [B, T, H, V] for output, [B, H, K, V] for state
        assert out.shape == torch.Size([B, T, H, V]), (
            f"Expected output shape [B,T,H,V]={torch.Size([B, T, H, V])}, got {out.shape}"
        )
        assert state is not None
        assert state.shape == torch.Size([B, H, K, V]), (
            f"Expected state shape [B,H,K,V]={torch.Size([B, H, K, V])}, got {state.shape}"
        )

        # Verify dtype: should be float32 (default Taichi dtype)
        assert out.dtype == torch.float32, (
            f"Expected output dtype float32, got {out.dtype}"
        )
        assert state.dtype == torch.float32, (
            f"Expected state dtype float32, got {state.dtype}"
        )

    def test_signature_matches_torch_reference(self):
        """The public op signature accepts the same args as the torch reference."""
        # Check recurrent_taichi_gdn2 signature matches torch reference
        public_sig = inspect.signature(recurrent_taichi_gdn2)
        public_params = set(public_sig.parameters.keys())

        # The public op should accept: q, k, v, g, b, w, initial_state, output_final_state, use_qk_l2norm_in_kernel
        required_params = {
            "q",
            "k",
            "v",
            "g",
            "b",
            "w",
            "initial_state",
            "output_final_state",
            "use_qk_l2norm_in_kernel",
        }
        assert required_params.issubset(public_params), (
            f"Missing params in recurrent_taichi_gdn2: {required_params - public_params}"
        )

        # Check chunk_taichi_gdn2 signature
        chunk_sig = inspect.signature(chunk_taichi_gdn2)
        chunk_params = set(chunk_sig.parameters.keys())

        chunk_required = {
            "q",
            "k",
            "v",
            "g",
            "b",
            "w",
            "initial_state",
            "output_final_state",
            "use_qk_l2norm_in_kernel",
            "chunk_size",
        }
        assert chunk_required.issubset(chunk_params), (
            f"Missing params in chunk_taichi_gdn2: {chunk_required - chunk_params}"
        )


# ---------------------------------------------------------------------------
# opcheck battery (mirrors the official Python custom-ops tutorial)
# ---------------------------------------------------------------------------
def _gdn2_inputs(
    B: int, T: int, H: int, K: int, V: int
) -> tuple[torch.Tensor | None, ...]:
    """Build a fresh input tuple for the GDN-2 ops (last slot is initial_state or None)."""
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = torch.zeros(B, T, H, K)
    b = torch.rand(B, T, H, K)
    w = torch.rand(B, T, H, V)
    return (q, k, v, g, b, w, None)


_GDN2_OPS = [
    ("chunk_gdn2", False),
    ("chunk_gdn2_with_state", True),
    ("recurrent_gdn2", False),
    ("recurrent_gdn2_with_state", True),
]


@pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)
@pytest.mark.parametrize("op_name,with_state", _GDN2_OPS)
def test_opcheck_battery(op_name: str, with_state: bool) -> None:
    """Each registered GDN-2 op passes ``opcheck`` on a battery of inputs.

    The battery covers the shapes the official Python custom-ops tutorial
    recommends: contiguous, empty, double dtype, and non-contiguous
    (swapped strides) inputs.
    """
    op = getattr(torch.ops.qwendopamine, op_name)
    base = _gdn2_inputs(1, 2, 2, 4, 4)
    examples: list[tuple[torch.Tensor | None, ...]] = [
        base,
        _gdn2_inputs(1, 0, 2, 4, 4),  # empty T axis
        tuple(
            t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
            for t in base
        ),  # fp64
    ]
    # Non-contiguous q via swapped inner strides.
    _q, k, v, g, b, w, init = base
    q_max = (1 - 1) * (2 * 2 * 4) + (2 - 1) * 4 + (2 - 1) * (2 * 4) + (4 - 1)
    q_storage = torch.empty(q_max + 1)
    q_t = q_storage.as_strided(
        size=(1, 2, 2, 4),
        stride=(2 * 2 * 4, 4, 2 * 4, 1),  # swap H and K dim strides
    )
    assert not q_t.is_contiguous()
    examples.append((q_t, k, v, g, b, w, init))

    for example in examples:
        opcheck(op, example, {})
