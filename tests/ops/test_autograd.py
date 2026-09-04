"""Autograd tests for the public ops.

Verifies that the autograd rules registered via
:mod:`qwendopamine.integrations.pytorch.autograd` propagate gradients
correctly through the public ops, and that ``torch.autograd.gradcheck``
passes on the readable references (smaller tensors because gradcheck is
exponential in input size).
"""

from __future__ import annotations

import torch

from qwendopamine.integrations.pytorch import (
    is_autograd_registered,
    is_registered,
)


# ---------------------------------------------------------------------------
# Custom-op registration sanity
# ---------------------------------------------------------------------------
def test_custom_ops_registered() -> None:
    """The public ops must be registered with ``torch.ops.qwendopamine``."""
    assert is_registered(), "Public ops not registered with torch.ops"
    ns = torch.ops.qwendopamine
    assert hasattr(ns, "chunk_gdn2")
    assert hasattr(ns, "recurrent_gdn2")
    assert hasattr(ns, "delta_core_step")


def test_custom_ops_have_autograd_rules() -> None:
    """Every public op must have an autograd rule attached."""
    assert is_autograd_registered(), "Public ops missing autograd rules"


# ---------------------------------------------------------------------------
# torch.autograd.gradcheck on the readable references
# ---------------------------------------------------------------------------
def test_reward_reference_step_gradcheck() -> None:
    """``torch.autograd.gradcheck`` must pass on the reward reference.

    Uses float64 for numerical stability (gradcheck requires
    high-precision gradients).
    """
    from qwendopamine.ops.references.reward_reference import (
        reward_reference_step,
    )

    torch.manual_seed(0)
    B, D = 1, 3
    state = torch.randn(B, D, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    omega_w = torch.zeros(B, 1, dtype=torch.float64, requires_grad=True)
    omega_e = torch.zeros(B, 1, dtype=torch.float64, requires_grad=True)

    # gradcheck needs a scalar output; sum the S_next output.
    def f(s, kv, v, ow, oe):
        return reward_reference_step(s, kv, v, ow, oe).sum()

    torch.autograd.gradcheck(
        f,
        (state, k, v, omega_w, omega_e),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


# ---------------------------------------------------------------------------
# Custom-op autograd correctness (numerical, not gradcheck)
# ---------------------------------------------------------------------------
def test_chunk_gdn2_python_op_backward_produces_finite_grads() -> None:
    """The production Python op (carrying the autograd Function) must
    produce finite gradients through the registered VJP.

    Note: the custom op ``torch.ops.qwendopamine.chunk_gdn2`` is a
    functional version for ``torch.compile``/``opcheck``; autograd
    goes through the ``qwendopamine.ops`` Python wrappers which
    already register the right ``torch.autograd.Function``.
    """
    from qwendopamine.ops import chunk_taichi_gdn2

    B, T, H, K, V = 1, 2, 2, 4, 4
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, requires_grad=True)
    k = torch.randn(B, T, H, K, requires_grad=True)
    v = torch.randn(B, T, H, V, requires_grad=True)
    g = torch.zeros(B, T, H, K, requires_grad=True)
    b = torch.rand(B, T, H, K, requires_grad=True)
    w = torch.rand(B, T, H, V, requires_grad=True)

    out, _ = chunk_taichi_gdn2(
        q=q, k=k, v=v, g=g, b=b, w=w, initial_state=None, output_final_state=True
    )
    out.sum().backward()
    for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("b", b), ("w", w)]:
        assert t.grad is not None, f"grad missing for {name}"
        assert torch.isfinite(t.grad).all(), f"grad NaN/Inf for {name}"


def test_delta_core_step_python_op_backward_produces_finite_grads() -> None:
    """The Reinforced Delta op's autograd must produce finite gradients."""
    from qwendopamine.ops import delta_core_step_out

    B, D = 1, 3
    torch.manual_seed(0)
    state = torch.randn(B, D, D, requires_grad=True)
    k = torch.randn(B, D, requires_grad=True)
    v = torch.randn(B, D, requires_grad=True)
    omega_w = torch.zeros(B, 1, requires_grad=True)
    omega_e = torch.zeros(B, 1, requires_grad=True)
    write = torch.rand(B, D, requires_grad=True)
    erase = torch.rand(B, D, requires_grad=True)
    next_state = torch.empty_like(state)
    out = delta_core_step_out(state, k, v, omega_w, omega_e, write, erase, next_state)
    out.sum().backward()
    for name, t in [
        ("state", state),
        ("k", k),
        ("v", v),
        ("omega_w", omega_w),
        ("omega_e", omega_e),
        ("write", write),
        ("erase", erase),
    ]:
        assert t.grad is not None, f"grad missing for {name}"
        assert torch.isfinite(t.grad).all(), f"grad NaN/Inf for {name}"
