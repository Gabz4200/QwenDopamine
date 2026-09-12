"""Smoke test for Taichi GDN-2 and delta ops via the public ops layer."""

import torch

from qwendopamine.kernels.taichi import is_available
from qwendopamine.ops.gdn2 import chunk_taichi_gdn2, recurrent_taichi_gdn2
from qwendopamine.ops.reward import delta_core_step


def _tiny_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    B, T, H, K, V = 1, 4, 1, 4, 4
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = torch.randn(B, T, H, K)
    b = torch.randn(B, T, H, K)
    w = torch.randn(B, T, H, V)
    return q, k, v, g, b, w


def test_when_taichi_ops_called_then_outputs_finite() -> None:
    if not is_available():
        return
    q, k, v, g, b, w = _tiny_inputs()
    out_c, _ = chunk_taichi_gdn2(q, k, v, g, b, w, output_final_state=False)
    out_r, _ = recurrent_taichi_gdn2(q, k, v, g, b, w, output_final_state=False)
    assert torch.isfinite(out_c).all()
    assert torch.isfinite(out_r).all()
    assert out_c.shape == (1, 4, 1, 4)


def test_when_delta_core_step_called_then_output_finite() -> None:
    if not is_available():
        return
    torch.manual_seed(1)
    state = torch.zeros(1, 4, 4)
    k = torch.randn(1, 4)
    v = torch.randn(1, 4)
    omega_w = torch.zeros(1, 1)
    omega_e = torch.zeros(1, 1)
    write = torch.rand(1, 4)
    erase = torch.rand(1, 4)
    out = delta_core_step(state, k, v, omega_w, omega_e, write, erase)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 4, 4)
