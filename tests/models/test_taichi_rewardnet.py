"""Numerical correctness tests for the Taichi Reinforced Delta kernels."""

from __future__ import annotations

import pytest
import torch

from qwendopamine.kernels.taichi import is_available
from qwendopamine.kernels.taichi.reinforced_kernels import launch_delta_core_step
from qwendopamine.models.reinforced.delta import DeltaMemoryCore

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)


def _spec_update(
    S: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w_eff: torch.Tensor,
    omega_e_eff: torch.Tensor,
) -> torch.Tensor:
    """Hand-derived column-wise spec matching ``DeltaMemoryCore._update_dense``.

    The state is laid out as ``S[b, d, k]`` (row=d, col=k). The torch
    reference computes
    ``outer[b, d, k] = e[b, d] * k[b, k]`` via
    ``torch.bmm(e.unsqueeze(-1), k.unsqueeze(1))``, so the rank-1
    update is ``omega_W * e[d] * k[k]``. The previous spec used
    ``k.unsqueeze(-1) * w_term.unsqueeze(1)`` which produced
    ``outer[b, j, k] = k[j] * w_term[k]`` — the axes were swapped,
    which made the spec the transpose of the torch reference and
    masked the kernel bug at loose tolerance.
    """
    e = v - (S @ k.unsqueeze(-1)).squeeze(-1)
    w_term = omega_w_eff * e
    return (1.0 - omega_e_eff).unsqueeze(-1) * S + w_term.unsqueeze(-1) * k.unsqueeze(1)


@pytest.mark.parametrize("B,D,T", [(1, 8, 4), (2, 8, 6), (1, 16, 3)])
def test_delta_core_step_matches_spec(B, D, T) -> None:
    torch.manual_seed(0)
    core = DeltaMemoryCore(d_model=D, use_short_conv=False, memory_rank=None)
    core.eval()
    x = torch.randn(B, T, D)
    S_ta = torch.zeros(B, D, D)
    ns = torch.empty_like(S_ta)
    S_ref = torch.zeros(B, D, D)
    for t in range(T):
        x_t = x[:, t, :]
        k_t = core.k_proj(x_t)
        v_t = core.v_proj(x_t)
        W = core.w_proj(x_t).sigmoid()
        E = core.e_proj(x_t).sigmoid()
        p = torch.full((B, 1), 0.5)
        w = torch.full((B, 1), 0.7)
        e = torch.full((B, 1), 0.3)
        ow = (p * w).squeeze(-1)
        oe = (p * e).squeeze(-1)
        omega_w_eff = (ow.unsqueeze(-1) * W).contiguous()
        omega_e_eff = (oe.unsqueeze(-1) * E).contiguous()
        launch_delta_core_step(S_ta, k_t, v_t, ow, oe, E, W, ns)
        S_ta, ns = ns, S_ta
        S_ref = _spec_update(S_ref, k_t, v_t, omega_w_eff, omega_e_eff)
    torch.testing.assert_close(S_ta, S_ref, atol=1e-5, rtol=1e-5)
