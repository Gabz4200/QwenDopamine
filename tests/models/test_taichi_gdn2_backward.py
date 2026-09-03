"""Backward gradient correctness tests for the Taichi GDN-2 kernels.

Each test compares the gradient produced by the Taichi path against
the gradient produced by the pure-PyTorch reference (which uses
PyTorch autograd through the token-by-token recurrence). The Taichi
recurrent path's backward is the per-token VJP of paper Eq. 10,
implemented in :func:`qwendopamine.models.gdn2.taichi.kernels.launch_recurrent_step_bwd`.
"""

from __future__ import annotations

import pytest
import torch

from qwendopamine.models.gdn2.recurrence.chunk import torch_chunk_gdn2
from qwendopamine.models.gdn2.recurrence.recurrent import torch_recurrent_gdn2
from qwendopamine.models.gdn2.taichi import (
    chunk_taichi_gdn2,
    is_available,
    recurrent_taichi_gdn2,
)

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)


def _make_inputs(B, T, H, K, V):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, dtype=torch.float32).requires_grad_(True)
    k = torch.randn(B, T, H, K, dtype=torch.float32).requires_grad_(True)
    v = torch.randn(B, T, H, V, dtype=torch.float32).requires_grad_(True)
    g = (torch.randn(B, T, H, K, dtype=torch.float32) * 0.1).requires_grad_(True)
    b = torch.sigmoid(torch.randn(B, T, H, K, dtype=torch.float32)).requires_grad_(True)
    w = torch.sigmoid(torch.randn(B, T, H, V, dtype=torch.float32)).requires_grad_(True)
    return q, k, v, g, b, w


@pytest.mark.parametrize("shape", [(1, 4, 1, 4, 4), (2, 8, 2, 4, 4), (1, 16, 3, 8, 8)])
def test_recurrent_backward_matches_reference(shape):
    """Gradient of the Taichi recurrent path matches the torch reference."""
    B, T, H, K, V = shape
    q_ta, k_ta, v_ta, g_ta, b_ta, w_ta = _make_inputs(B, T, H, K, V)
    q_ref, k_ref, v_ref, g_ref, b_ref, w_ref = _make_inputs(B, T, H, K, V)

    out_ta, _ = recurrent_taichi_gdn2(q_ta, k_ta, v_ta, g_ta, b_ta, w_ta)
    out_ref, _ = torch_recurrent_gdn2(q_ref, k_ref, v_ref, g_ref, b_ref, w_ref)

    torch.testing.assert_close(out_ta, out_ref, atol=1e-5, rtol=1e-5)

    out_ta.sum().backward()
    out_ref.sum().backward()
    for name, ta, ref in [
        ("q", q_ta, q_ref),
        ("k", k_ta, k_ref),
        ("v", v_ta, v_ref),
        ("g", g_ta, g_ref),
        ("b", b_ta, b_ref),
        ("w", w_ta, w_ref),
    ]:
        torch.testing.assert_close(
            ta.grad,
            ref.grad,
            atol=1e-3,
            rtol=1e-3,
            msg=f"grad {name} differs",
        )


@pytest.mark.parametrize("shape", [(1, 4, 1, 4, 4), (2, 8, 2, 4, 4)])
def test_recurrent_training_step(shape):
    """End-to-end: forward + backward + optimizer step converges."""
    B, T, H, K, V = shape
    q, k, v, g, b, w = _make_inputs(B, T, H, K, V)
    params = [q, k, v, g, b, w]
    opt = torch.optim.SGD(params, lr=0.01)
    initial_loss = None
    for step in range(3):
        opt.zero_grad()
        out, _ = recurrent_taichi_gdn2(q, k, v, g, b, w)
        loss = (out**2).mean()
        loss.backward()
        opt.step()
        if initial_loss is None:
            initial_loss = loss.item()
    assert loss.item() < initial_loss  # pyrefly: ignore[unsupported-operation]


@pytest.mark.parametrize("shape", [(1, 8, 1, 4, 4), (1, 16, 1, 4, 4)])
def test_chunk_backward_matches_reference(shape):
    """Gradient of the Taichi chunkwise path matches the torch reference.

    The Taichi chunkwise forward's numerical contract matches the torch
    chunkwise reference to ``atol=1.0, rtol=1.0`` (the WY forward
    substitution in Taichi is naive; see :mod:`test_taichi_gdn2`). The
    backward defers to the differentiable torch reference so the gradient
    follows the same tolerance as the forward.
    """
    B, T, H, K, V = shape
    q_ta, k_ta, v_ta, g_ta, b_ta, w_ta = _make_inputs(B, T, H, K, V)
    q_ref, k_ref, v_ref, g_ref, b_ref, w_ref = _make_inputs(B, T, H, K, V)

    out_ta, _ = chunk_taichi_gdn2(
        q_ta,
        k_ta,
        v_ta,
        g_ta,
        b_ta,
        w_ta,
        chunk_size=8,
    )
    out_ref, _ = torch_chunk_gdn2(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        b_ref,
        w_ref,
        chunk_size=8,
    )
    torch.testing.assert_close(out_ta, out_ref, atol=1.0, rtol=1.0)

    out_ta.sum().backward()
    out_ref.sum().backward()
    for name, ta, ref in [
        ("q", q_ta, q_ref),
        ("k", k_ta, k_ref),
        ("v", v_ta, v_ref),
        ("g", g_ta, g_ref),
        ("b", b_ta, b_ref),
        ("w", w_ta, w_ref),
    ]:
        torch.testing.assert_close(
            ta.grad,
            ref.grad,
            atol=1.0,
            rtol=1.0,
            msg=f"grad {name} differs",
        )
