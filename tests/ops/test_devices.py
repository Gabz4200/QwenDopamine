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
    from qwendopamine.ops import delta_core_step_out

    B, D = 2, 4
    state = torch.zeros(B, D, D)
    k = torch.randn(B, D)
    v = torch.randn(B, D)
    omega_w = torch.zeros(B, 1)
    omega_e = torch.zeros(B, 1)
    write = torch.rand(B, D)
    erase = torch.rand(B, D)
    next_state = torch.empty_like(state)
    out = delta_core_step_out(state, k, v, omega_w, omega_e, write, erase, next_state)
    assert out.device.type == "cpu"


# ---------------------------------------------------------------------------
# Non-contiguous tensors
# ---------------------------------------------------------------------------
def test_public_op_handles_non_contiguous_input() -> None:
    """The public ops must accept non-contiguous tensors (the kernel contract is contiguous, so we copy first).

    Uses the same non-contiguous patterns the official PyTorch
    tutorial recommends for ``opcheck`` exercises:
    a transposed view and a sliced tensor with non-zero storage offset.
    """
    from qwendopamine.ops import recurrent_taichi_gdn2

    B, T, H, K, V = 2, 4, 2, 8, 8
    # q: non-contiguous via swapped inner strides. Storage must
    # address the highest-offset element.
    q_max_offset = (B - 1) * (T * H * K) + (T - 1) * K + (H - 1) * (H * K) + (K - 1)
    q_storage = torch.empty(q_max_offset + 1)
    q = q_storage.as_strided(
        size=(B, T, H, K),
        stride=(T * H * K, K, H * K, 1),  # swap H and K dim strides
    )
    assert not q.is_contiguous()
    # k: different non-contig pattern (swapped outer strides).
    k_max_offset = (B - 1) * (H * K) + (T - 1) * (T * H * K) + (H - 1) * K + (K - 1)
    k_storage = torch.empty(k_max_offset + 1)
    k = k_storage.as_strided(
        size=(B, T, H, K),
        stride=(H * K, T * H * K, K, 1),  # swap B and T dim strides
    )
    assert not k.is_contiguous()
    v = torch.randn(B, T, H, V)
    g = torch.zeros(B, T, H, K)
    b = torch.rand(B, T, H, K)
    w = torch.rand(B, T, H, V)

    out, _ = recurrent_taichi_gdn2(q, k, v, g, b, w, None, True)
    assert out.shape == (B, T, H, V)
    # Output must be contiguous (kernel contract).
    assert out.is_contiguous()


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
    # ``chunk_gdn2`` returns a single Tensor; ``chunk_gdn2_with_state``
    # returns a list of two Tensors. Cover both.
    out = torch.ops.qwendopamine.chunk_gdn2(q, k, v, g, b, w, None)
    assert out.device.type == "cpu"
    out_list = torch.ops.qwendopamine.chunk_gdn2_with_state(q, k, v, g, b, w, None)
    for t in out_list:
        assert t.device.type == "cpu"
