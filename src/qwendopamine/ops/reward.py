"""Reinforced Delta public op with Taichi fallback.

Backend choice is delegated to Taichi: the kernel runtime picks CUDA →
Vulkan → Metal/OpenGL → CPU on its own. When Taichi is unavailable,
this module falls back to a pure-PyTorch implementation of the same
column-wise recurrence.

Two entry points are exposed:

- :func:`delta_core_step` — **functional** API. Returns a fresh
  ``next_state`` tensor and never writes into any input. This is the
  contract used by the public op, by autograd, and by ``torch.compile``.
- :func:`delta_core_step_out` — **in-place** API. Writes the result
  into the caller-supplied ``next_state`` buffer and returns it. The
  Taichi kernel uses this to avoid an allocation on every step. The
  PyTorch fallback is implemented by delegating to
  :func:`delta_core_step` and ``copy_``-ing into the destination.

Both paths use the same column-wise recurrence:

    e[d]        = v[d] - sum_kk S[d, kk] * k[kk]
    w_term[d]   = omega_w_eff * write[d]      with omega_w_eff = omega_w * write
    e_term[d]   = omega_e_eff * erase[d]      with omega_e_eff = omega_e * erase
    S_next[d,k] = (1 - e_term[d]) * S[d, k] + w_term[d] * e[d] * k[k]
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
    """Pure-PyTorch **functional** Reinforced Delta per-token update.

    Returns a fresh ``next_state`` tensor. The contract is "no aliasing,
    no in-place writes"; callers can use the result without worrying
    about accidentally mutating ``state``.
    """
    omega_w_eff = _make_effective_gate(omega_w, write)
    omega_e_eff = _make_effective_gate(omega_e, erase)
    e = v - (state @ k.unsqueeze(-1)).squeeze(-1)
    return (1.0 - omega_e_eff).unsqueeze(-1) * state + omega_w_eff.unsqueeze(-1) * (
        e.unsqueeze(-1) * k.unsqueeze(1)
    )


def delta_core_step(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
) -> torch.Tensor:
    """Reinforced Delta per-token update with backend delegation (functional).

    Both the Taichi kernel and the pure-PyTorch fallback are invoked
    purely functionally: a fresh ``next_state`` is allocated and
    returned. No input tensor is mutated.

    When Taichi is available, the autograd-aware kernel records the
    per-token adjoint. When Taichi is unavailable, the pure-PyTorch
    fallback below implements the same column-wise spec.
    """
    if _is_available():
        from qwendopamine.kernels.taichi import (
            delta_core_step as _taichi_delta_core_step,
        )

        return _taichi_delta_core_step(state, k, v, omega_w, omega_e, write, erase)

    return _reward_torch_step(state, k, v, omega_w, omega_e, write, erase)


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
    """Reinforced Delta per-token update with backend delegation (in-place).

    Writes the result into the caller-supplied ``next_state`` buffer
    and returns it. The Taichi kernel uses this to avoid an allocation
    on every step. The PyTorch fallback delegates to
    :func:`delta_core_step` (functional) and ``copy_``-es into the
    destination, so the two paths share the same numerical contract.

    Contract:
      - ``next_state`` must be allocated, on the same device as
        ``state``, and have the same shape and dtype.
      - ``next_state`` is mutated; the return value is the same
        tensor.
      - ``state`` is not mutated.
    """
    if _is_available():
        from qwendopamine.kernels.taichi import (
            delta_core_step_out as _taichi_delta_core_step_out,
        )

        return _taichi_delta_core_step_out(
            state, k, v, omega_w, omega_e, write, erase, next_state
        )

    next_state.copy_(delta_core_step(state, k, v, omega_w, omega_e, write, erase))
    return next_state
