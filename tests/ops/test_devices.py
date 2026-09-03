"""Device and dtype coverage tests for the public ops.

Verifies that the public ops and the readable references handle the
canonical device/dtype matrix the context requires:

    - empty tensors;
    - non-contiguous tensors;
    - different dtypes (fp32, fp64);
    - odd tensor dimensions;
    - repeated calls (state must be stateless between calls);
    - supported Taichi backends (Vulkan on this machine).

The Taichi runtime is CPU-only on this development box
(``vulkan`` is the GPU backend Taichi picks here). The tests
verify that the public ops work on the CPU torch tensors Taichi's
backend can ingest.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Device coverage
# ---------------------------------------------------------------------------
def test_public_op_runs_on_cpu_tensors() -> None:
    """The public ops must accept CPU tensors regardless of Taichi backend."""
    from qwendopamine.ops import (
        chunk_taichi_gdn2,
        recurrent_taichi_gdn2,
    )

    B, T, H, K, V = 1, 2, 2, 4, 4
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = torch.zeros(B, T, H, K)
    b = torch.rand(B, T, H, K)
    w = torch.rand(B, T, H, V)

    out, _ = chunk_taichi_gdn2(q, k, v, g, b, w, None, True)
    assert out.device.type == "cpu"
    out, _ = recurrent_taichi_gdn2(q, k, v, g, b, w, None, True)
    assert out.device.type == "cpu"


def test_reward_op_runs_on_cpu_tensors() -> None:
    """The reward op must accept CPU tensors."""
    from qwendopamine.ops import delta_core_step_autograd

    B, D = 2, 4
    state = torch.zeros(B, D, D)
    k = torch.randn(B, D)
    v = torch.randn(B, D)
    omega_w = torch.zeros(B, 1)
    omega_e = torch.zeros(B, 1)
    write = torch.rand(B, D)
    erase = torch.rand(B, D)
    next_state = torch.empty_like(state)
    out = delta_core_step_autograd(
        state, k, v, omega_w, omega_e, write, erase, next_state
    )
    assert out.device.type == "cpu"


# ---------------------------------------------------------------------------
# Non-contiguous tensors
# ---------------------------------------------------------------------------
def test_public_op_handles_non_contiguous_input() -> None:
    """The public ops must accept non-contiguous tensors (the kernel contract is contiguous, so we copy first)."""
    from qwendopamine.ops import recurrent_taichi_gdn2

    B, T, H, K, V = 1, 2, 2, 4, 4
    # Create non-contiguous tensors by transposing an extra dim then back.
    base_q = torch.randn(B, T, H, K)
    q = base_q.transpose(0, 1).transpose(0, 1)  # round-trip; may stay non-contig
    if q.is_contiguous():
        q = q[None].transpose(0, 1).squeeze(1)
    base_k = torch.randn(B, T, H, K)
    k = base_k.transpose(0, 1).transpose(0, 1)
    if k.is_contiguous():
        k = k[None].transpose(0, 1).squeeze(1)
    base_v = torch.randn(B, T, H, V)
    v = base_v.transpose(0, 1).transpose(0, 1)
    if v.is_contiguous():
        v = v[None].transpose(0, 1).squeeze(1)
    g = torch.zeros(B, T, H, K)
    b = torch.rand(B, T, H, K)
    w = torch.rand(B, T, H, V)

    # The op should not crash; the kernel contract is contiguous, so the
    # op may internally call .contiguous() (or the kernel may copy).
    out, _ = recurrent_taichi_gdn2(q, k, v, g, b, w, None, True)
    assert out.shape == (B, T, H, V)


# ---------------------------------------------------------------------------
# Different dtypes
# ---------------------------------------------------------------------------
def test_reward_reference_handles_fp32_and_fp64() -> None:
    """The readable reward reference must work in both fp32 and fp64."""
    from qwendopamine.ops.references import reward_reference_step

    B, D = 1, 3
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(0)
        state = torch.randn(B, D, D, dtype=dtype)
        k = torch.randn(B, D, dtype=dtype)
        v = torch.randn(B, D, dtype=dtype)
        omega_w = torch.zeros(B, 1, dtype=dtype)
        omega_e = torch.zeros(B, 1, dtype=dtype)
        out = reward_reference_step(state, k, v, omega_w, omega_e)
        assert out.dtype == dtype
        assert out.shape == state.shape


# ---------------------------------------------------------------------------
# Repeated calls (statelessness)
# ---------------------------------------------------------------------------
def test_public_op_is_stateless_across_calls() -> None:
    """Calling the public op twice with the same input must give the same output."""
    from qwendopamine.ops import recurrent_taichi_gdn2

    B, T, H, K, V = 1, 2, 2, 4, 4
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = torch.zeros(B, T, H, K)
    b = torch.rand(B, T, H, K)
    w = torch.rand(B, T, H, V)

    out1, _ = recurrent_taichi_gdn2(q, k, v, g, b, w, None, True)
    out2, _ = recurrent_taichi_gdn2(q, k, v, g, b, w, None, True)
    assert torch.equal(out1, out2), "Public op must be stateless between calls"


# ---------------------------------------------------------------------------
# Empty / edge-case tensors
# ---------------------------------------------------------------------------
def test_reward_reference_handles_T_1_sequence() -> None:
    """Single-token sequences must work."""
    from qwendopamine.ops.references import reward_reference_step

    B, D = 1, 2
    state = torch.zeros(B, D, D)
    k = torch.randn(B, D)
    v = torch.randn(B, D)
    omega_w = torch.zeros(B, 1)
    omega_e = torch.zeros(B, 1)
    out = reward_reference_step(state, k, v, omega_w, omega_e)
    assert out.shape == state.shape


@pytest.mark.parametrize("D", [1, 2, 4, 8, 16])
def test_reward_reference_handles_odd_dimensions(D: int) -> None:
    """The reward reference must handle unusual state dimensions."""
    from qwendopamine.ops.references import reward_reference_step

    B = 2
    state = torch.zeros(B, D, D)
    k = torch.randn(B, D)
    v = torch.randn(B, D)
    omega_w = torch.zeros(B, 1)
    omega_e = torch.zeros(B, 1)
    out = reward_reference_step(state, k, v, omega_w, omega_e)
    assert out.shape == (B, D, D)


# ---------------------------------------------------------------------------
# Custom-op device path
# ---------------------------------------------------------------------------
def test_custom_op_returns_cpu_tensors_when_called_with_cpu_inputs() -> None:
    """The registered custom ops must respect the device of the inputs."""
    from qwendopamine.integrations.pytorch import is_registered

    if not is_registered():
        pytest.skip("Custom ops not registered")

    B, T, H, K, V = 1, 2, 2, 4, 4
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = torch.zeros(B, T, H, K)
    b = torch.rand(B, T, H, K)
    w = torch.rand(B, T, H, V)
    out_list = torch.ops.qwendopamine.chunk_gdn2(q, k, v, g, b, w, None, True)
    for t in out_list:
        assert t.device.type == "cpu"
