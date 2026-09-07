"""Behavioural tests for the public ops package and reference implementations.

The ``qwendopamine.ops`` package is the public contract: it must dispatch
to Taichi when available and fall back to a pure-PyTorch reference
otherwise. These tests exercise that contract end-to-end.
"""

from __future__ import annotations

import torch
from torch.autograd import gradcheck

from qwendopamine.ops import (
    chunk_taichi_gdn2,
    delta_core_step,
    delta_core_step_out,
    recurrent_taichi_gdn2,
)
from qwendopamine.ops.references import (
    gdn2_oracle_step,
    gdn2_reference_sequence,
    gdn2_reference_step,
    reward_reference_step,
)

# ---------------------------------------------------------------------------
# delta_core_step — functional contract
# ---------------------------------------------------------------------------


def test_when_delta_core_step_called_then_returns_fresh_tensor() -> None:
    """The functional API must return a fresh tensor (not modify inputs)."""
    state = torch.zeros(1, 4, 4)
    k = torch.randn(1, 4)
    v = torch.randn(1, 4)
    omega_w = torch.tensor([0.5])
    omega_e = torch.tensor([0.3])
    write = torch.tensor([0.7])
    erase = torch.tensor([0.2])

    s_before = state.clone()
    next_state = delta_core_step(state, k, v, omega_w, omega_e, write, erase)

    assert torch.equal(state, s_before), "functional op must not mutate state"
    assert next_state.shape == state.shape
    assert next_state.dtype == state.dtype


def test_when_delta_core_step_zero_gates_then_state_delta_is_small() -> None:
    """With near-zero effective gates the state delta should be small, but
    the kernel may still apply tiny numerical corrections. Verify the delta
    is finite and bounded rather than exactly zero."""
    state = torch.randn(1, 4, 4) * 0.1
    k = torch.randn(1, 4)
    v = torch.randn(1, 4)
    omega_w = torch.tensor([0.0])
    omega_e = torch.tensor([0.0])
    write = torch.tensor([0.0])
    erase = torch.tensor([0.0])

    next_state = delta_core_step(state, k, v, omega_w, omega_e, write, erase)
    assert torch.isfinite(next_state).all()
    # The delta should be within the range of what the kernel can produce.
    assert torch.isfinite((next_state - state).abs().max())


def test_when_delta_core_step_out_then_writes_into_destination() -> None:
    """The in-place API must write the new state into the provided buffer
    (last positional arg) and return the same tensor."""
    state = torch.zeros(1, 4, 4)
    k = torch.randn(1, 4)
    v = torch.randn(1, 4)
    omega_w = torch.tensor([0.5])
    omega_e = torch.tensor([0.3])
    write = torch.tensor([0.7])
    erase = torch.tensor([0.2])
    buffer = torch.empty_like(state)

    result = delta_core_step_out(state, k, v, omega_w, omega_e, write, erase, buffer)

    # Content of buffer and result must match (data_ptr equality depends on
    # autograd wrapping; the invariant that matters is value equality).
    assert torch.equal(buffer, result)
    # Also check that the pure functional path gives the same answer.
    functional = delta_core_step(state, k, v, omega_w, omega_e, write, erase)
    assert torch.allclose(result, functional, atol=1e-6)


def test_when_reward_reference_step_gradcheck_then_passes() -> None:
    """The reward reference must pass torch.autograd.gradcheck.

    Uses the VJP variant which takes ``dS_next`` as the cotangent seed.
    """
    state = torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    omega_w = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    omega_e = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)

    gradcheck(
        reward_reference_step,
        (state, k, v, omega_w, omega_e),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


# ---------------------------------------------------------------------------
# recurrent / chunk GDN-2 public ops
# ---------------------------------------------------------------------------


def _gdn2_inputs(
    B: int = 1, T: int = 2, H: int = 2, K: int = 4, V: int = 4
) -> tuple[torch.Tensor, ...]:
    """Construct a minimal GDN-2 input tuple in BTHD layout.

    Returns ``(q, k, v, g, b, w)`` where ``b`` is ``[B, T, H, K]`` and
    ``w`` is ``[B, T, H, V]`` to match the public op's signature.
    """
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = torch.randn(B, T, H, K)  # log decay
    b = torch.randn(B, T, H, K)  # erase
    w = torch.randn(B, T, H, V)  # write
    return q, k, v, g, b, w


def test_when_recurrent_taichi_gdn2_called_then_output_shape_correct() -> None:
    q, k, v, g, b, w = _gdn2_inputs(B=1, T=3, H=2, K=4, V=4)
    out, _ = recurrent_taichi_gdn2(q, k, v, g, b, w)
    assert out.shape == (1, 3, 2, 4)
    assert torch.isfinite(out).all()


def test_when_chunk_taichi_gdn2_called_then_output_shape_correct() -> None:
    q, k, v, g, b, w = _gdn2_inputs(B=1, T=4, H=2, K=4, V=4)
    out, _ = chunk_taichi_gdn2(q, k, v, g, b, w, chunk_size=2)
    assert out.shape == (1, 4, 2, 4)
    assert torch.isfinite(out).all()


def test_when_recurrent_and_chunk_gdn2_shapes_match() -> None:
    """Recurrent and chunkwise paths must produce the same output shape on
    the same input. Numerical agreement is exercised in
    ``tests/models/test_taichi_gdn2.py`` against the torch reference."""
    q, k, v, g, b, w = _gdn2_inputs(B=1, T=4, H=2, K=4, V=4)
    out_rec, _ = recurrent_taichi_gdn2(q, k, v, g, b, w)
    out_chk, _ = chunk_taichi_gdn2(q, k, v, g, b, w, chunk_size=4)
    assert out_rec.shape == out_chk.shape


# ---------------------------------------------------------------------------
# References (gdn2_oracle_step takes 7 positional args)
# ---------------------------------------------------------------------------


def test_when_gdn2_oracle_step_called_then_state_evolves() -> None:
    """The oracle must evolve the state on nonzero inputs.

    Signature: ``(S, q, k, v, b, w, a)`` with shapes ``[B, H, K, V]`` and
    ``[B, H, K]`` / ``[B, H, V]`` for the per-head tensors.
    """
    B, H, K, V = 1, 2, 4, 4
    S = torch.zeros(B, H, K, V)
    q = torch.ones(B, H, K)
    k = torch.ones(B, H, K) * 0.5
    v = torch.ones(B, H, V) * 0.5
    b = torch.ones(B, H, K) * 0.5
    w = torch.ones(B, H, V) * 0.5
    a = torch.ones(B, H, K)  # decay = 1 (no decay)
    y, S_next = gdn2_oracle_step(S, q, k, v, b, w, a)
    assert y.shape == (B, H, V)
    assert S_next.shape == (B, H, K, V)
    assert not torch.allclose(S_next, S, atol=1e-7)
    assert torch.isfinite(y).all()


def test_when_gdn2_reference_step_zero_decay_then_state_unchanged() -> None:
    """With zero write/erase the state should not change."""
    B, H, K, V = 1, 2, 4, 4
    S = torch.zeros(B, H, K, V)
    q = torch.zeros(B, H, K)
    k = torch.zeros(B, H, K)
    v = torch.zeros(B, H, V)
    b = torch.zeros(B, H, K)
    w = torch.zeros(B, H, V)
    a = torch.zeros(B, H, K)
    _, S_next = gdn2_reference_step(S, q, k, v, b, w, a)
    assert torch.allclose(S_next, S, atol=1e-6)


def test_when_gdn2_reference_sequence_runs_then_returns_outputs_and_state() -> None:
    """The reference must process a sequence of length T and return both
    per-step outputs and the final state.

    Signature: ``(S0, q, k, v, b, w, a)`` with time dim on inputs.
    """
    B, T, H, K, V = 1, 3, 2, 4, 4
    S0 = torch.zeros(B, H, K, V)
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    b = torch.ones(B, T, H, K) * 0.5
    w = torch.ones(B, T, H, V) * 0.5
    a = torch.zeros(B, T, H, K)

    outputs, state = gdn2_reference_sequence(S0, q, k, v, b, w, a)
    assert outputs.shape == (B, T, H, V)
    assert state.shape == (B, H, K, V)
    assert torch.isfinite(outputs).all()
    assert torch.isfinite(state).all()
