"""Reinforced Delta public op with Taichi fallback.

Backend choice is delegated to Taichi: the kernel runtime picks CUDA →
Vulkan → Metal/OpenGL → CPU on its own. When Taichi is unavailable,
this module falls back to a pure-PyTorch implementation of the same
column-wise recurrence.
"""

import torch

from qwendopamine.kernels.taichi import is_available as _is_available
from qwendopamine.kernels.taichi.reinforced_kernels import _make_effective_gate


def _reward_torch_step(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch functional Reinforced Delta per-token update.

    Returns a fresh ``next_state`` tensor. The same column-wise spec
    as the Taichi kernel:

        e[d] = v[d] - sum_kk S[d, kk] * k[kk]
        w_term[d] = omega_w * write[d]
        oe_term[d] = omega_e * erase[d]
        S_next[d, k] = (1 - oe_term[d]) * S[d, k] + w_term[d] * e[d] * k[k]
    """
    omega_w_eff = _make_effective_gate(omega_w, write)
    omega_e_eff = _make_effective_gate(omega_e, erase)
    e = v - (state @ k.unsqueeze(-1)).squeeze(-1)
    return (1.0 - omega_e_eff).unsqueeze(-1) * state + omega_w_eff.unsqueeze(-1) * (
        e.unsqueeze(-1) * k.unsqueeze(1)
    )


def delta_core_step_out(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
    next_state: torch.Tensor,
) -> torch.Tensor:
    """Reinforced Delta per-token update with backend delegation.

    When Taichi is available, the autograd-aware kernel records the
    per-token adjoint. When Taichi is unavailable, the pure-PyTorch
    fallback below implements the same column-wise spec.
    """
    if _is_available():
        from qwendopamine.kernels.taichi import delta_core_step_out as _fn

        return _fn(
            state,
            k,
            v,
            omega_w,
            omega_e,
            write,
            erase,
            next_state,
        )

    omega_w_eff = _make_effective_gate(omega_w, write)  # [B, D]
    omega_e_eff = _make_effective_gate(omega_e, erase)  # [B, D]

    # e = v - sum_{kk} S[d, kk] * k[kk]  (per-batch, per-dim residual)
    e = v - (state @ k.unsqueeze(-1)).squeeze(-1)  # [B, D]

    # S_next[d, k] = (1 - omega_e_eff[d]) * S[d, k] + omega_w_eff[d] * e[d] * k[k]
    next_state.copy_(
        (1.0 - omega_e_eff).unsqueeze(-1) * state
        + omega_w_eff.unsqueeze(-1) * (e.unsqueeze(-1) * k.unsqueeze(1))
    )

    return next_state
