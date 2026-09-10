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
    omega_w_eff[d] = omega_w[d] * write[d]      # per-channel effective write gate
    omega_e_eff[d] = omega_e[d] * erase[d]      # per-channel effective erase gate
    S_next[d,k] = (1 - omega_e_eff[d]) * S[d, k] + omega_w_eff[d] * e[d] * k[k]
"""

from __future__ import annotations

import torch

from qwendopamine.ops._backend_registry import register_backend, resolve_backend


def _make_effective_gate_local(omega: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Local effective gate helper to avoid importing kernels at module import time."""
    if omega.dim() == 1:
        omega = omega.unsqueeze(-1)
    if omega.dim() != 2 or omega.shape[-1] != 1:
        raise ValueError(f"omega must be [B] or [B, 1]; got shape {tuple(omega.shape)}")
    return (omega * gate).contiguous()


def _register_reward_backends() -> None:
    def _torch_backend() -> str:
        return "torch"

    def _taichi_backend() -> str:
        from qwendopamine.kernels.taichi import is_available as taichi_is_available

        return "taichi" if taichi_is_available() else "torch"

    register_backend("torch", _torch_backend)
    register_backend("taichi", _taichi_backend)
    register_backend("auto", _taichi_backend)


_register_reward_backends()


def is_taichi_available() -> bool:
    r"""Return whether Taichi Reinforced Delta backend is available."""
    from qwendopamine.kernels.taichi import is_available as _is_available

    return _is_available()


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
    omega_w_eff = _make_effective_gate_local(omega_w, write)
    omega_e_eff = _make_effective_gate_local(omega_e, erase)
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
    backend = resolve_backend("auto")
    if backend == "taichi":
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
    backend = resolve_backend("auto")
    if backend == "taichi":
        from qwendopamine.kernels.taichi import (
            delta_core_step_out as _taichi_delta_core_step_out,
        )

        return _taichi_delta_core_step_out(
            state, k, v, omega_w, omega_e, write, erase, next_state
        )

    next_state.copy_(delta_core_step(state, k, v, omega_w, omega_e, write, erase))
    return next_state


__all__ = [
    "delta_core_step",
    "delta_core_step_out",
]
