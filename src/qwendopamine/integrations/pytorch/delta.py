"""GDN-2 delta core step operations.

Implements the ``qwendopamine::delta_core_step`` custom op split into
body, fake, public API, and autograd modules.

Schema contract
---------------
Each op follows the "stable schema" rule from the
`Custom Python Operators tutorial
<https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html>`_:

    - the number of return Tensors is fixed per op (no maybe-out);
    - the returned Tensors do not alias any input Tensor;
    - the fake kernel matches the real kernel's output metadata
      (shape, dtype, device, layout, strides, storage offset);
    - the fake kernel may inspect metadata but must not read data.

Autograd contract
-----------------
The delta_core_step op supports a functional backward via
``register_autograd(setup_context, backward)``.  The setup function
saves tensors for the backward pass and returns a dummy output tensor.
The backward computes hand-derived VJP gradients matching the column-wise
Reinforced Delta update rule.

    Forward (shapes in parens):
        omega_w_eff [B, D]  = omega_w [B, 1] * write [B, D]
        omega_e_eff [B, D]  = omega_e [B, 1] * erase [B, D]
        e [B, D]            = v [B, D] - state [B, D, D] @ k [B, D]
        e_term [B, D, D]    = e [B, D, 1] * k [B, 1, D]
        next_state [B, D, D] = (1 - omega_e_eff) [B, D, 1] * state [B, D, D]
                             + omega_w_eff [B, D, 1] * e_term [B, D, D]

    Backward (with g = grad_out of shape [B, D, D]):
        d_e_term [B, D, D] = g * omega_w_eff[B, D, 1]
        d_e [B, D]         = (d_e_term * k[B, 1, D]).sum(dim=-1)
        d_state [B, D, D]  = g * (1 - omega_e_eff)[B, D, 1] - d_e[B, D, 1] * k[B, 1, D]
        d_k [B, D]         = (d_e_term * e[B, D, 1]).sum(dim=1)
        d_v [B, D]         = d_e
        d_omega_w_eff [B, D] = (g * e_term).sum(dim=-1)
        d_omega_w [B, 1]   = (d_omega_w_eff * write).sum(dim=-1, keepdim=True)
        d_write [B, D]     = d_omega_w_eff * omega_w
        d_omega_e_eff [B, D] = -(g * state).sum(dim=-1)
        d_omega_e [B, 1]   = (d_omega_e_eff * erase).sum(dim=-1, keepdim=True)
        d_erase [B, D]     = d_omega_e_eff * omega_e

    All math runs in float32 for numerical stability on fp16 inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qwendopamine.integrations.pytorch.register import _route_to_active_device

if TYPE_CHECKING:
    pass
else:
    from torch.library import custom_op  # runtime import for decorators


def _delta_core_step_body(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
) -> torch.Tensor:
    from qwendopamine.ops.reward import _reward_torch_step

    return _reward_torch_step(state, k, v, omega_w, omega_e, write, erase).contiguous()


@custom_op(
    "qwendopamine::delta_core_step",
    mutates_args=(),
    device_types="cpu",
)
def delta_core_step_op(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
) -> torch.Tensor:
    """Reinforced Delta per-token update; returns a fresh ``next_state``.

    Functional: never writes into ``state`` and never aliases it. The
    in-place variant ``qwendopamine::delta_core_step_out`` (in
    :mod:`qwendopamine.ops.reward`) writes into a caller-provided
    buffer; this functional version exists for ``torch.compile`` and
    ``register_autograd`` compatibility.
    """
    result = _route_to_active_device(
        _delta_core_step_body, state, k, v, omega_w, omega_e, write, erase
    )
    assert isinstance(result, torch.Tensor)
    return result


@delta_core_step_op.register_fake
def _delta_core_step_fake(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation. Same shape, dtype, and device as ``state``."""
    return torch.empty_like(state)


def _delta_core_step_out_setup(
    ctx: torch.autograd.function.FunctionCtx,
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
) -> torch.Tensor:
    """Save inputs and re-compute the per-step output for the backward pass."""
    from qwendopamine.ops.reward import _reward_torch_step

    out = _reward_torch_step(state, k, v, omega_w, omega_e, write, erase)
    ctx.save_for_backward(state, k, v, omega_w, omega_e, write, erase, out)
    return out


def _delta_core_step_out_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Hand-derived VJP for the column-wise Reinforced Delta update.

    Forward (shapes in parens):
        omega_w_eff [B, D]  = omega_w [B, 1] * write [B, D]
        omega_e_eff [B, D]  = omega_e [B, 1] * erase [B, D]
        e [B, D]            = v [B, D] - state [B, D, D] @ k [B, D]
        e_term [B, D, D]    = e [B, D, 1] * k [B, 1, D]
        next_state [B, D, D] = (1 - omega_e_eff) [B, D, 1] * state [B, D, D]
                             + omega_w_eff [B, D, 1] * e_term [B, D, D]

    Backward (with g = grad_out of shape [B, D, D]):
        d_e_term [B, D, D] = g * omega_w_eff[B, D, 1]
        d_e [B, D]         = (d_e_term * k[B, 1, D]).sum(dim=-1)
        d_state [B, D, D]  = g * (1 - omega_e_eff)[B, D, 1] - d_e[B, D, 1] * k[B, 1, D]
        d_k [B, D]         = (d_e_term * e[B, D, 1]).sum(dim=1)
        d_v [B, D]         = d_e
        d_omega_w_eff [B, D] = (g * e_term).sum(dim=-1)
        d_omega_w [B, 1]   = (d_omega_w_eff * write).sum(dim=-1, keepdim=True)
        d_write [B, D]     = d_omega_w_eff * omega_w
        d_omega_e_eff [B, D] = -(g * state).sum(dim=-1)
        d_omega_e [B, 1]   = (d_omega_e_eff * erase).sum(dim=-1, keepdim=True)
        d_erase [B, D]     = d_omega_e_eff * omega_e

    All math runs in float32 for numerical stability on fp16 inputs.
    """
    state, k, v, omega_w, omega_e, write, erase, _out = ctx.saved_tensors  # type: ignore[attr-defined]

    state32 = state.float()
    k32 = k.float()
    v32 = v.float()
    omega_w32 = omega_w.float()
    omega_e32 = omega_e.float()
    write32 = write.float()
    erase32 = erase.float()
    g = grad_out.float()

    omega_w_eff = (omega_w32 * write32).unsqueeze(-1)  # [B, D, 1]
    omega_e_eff = (omega_e32 * erase32).unsqueeze(-1)  # [B, D, 1]
    e = v32 - (state32 @ k32.unsqueeze(-1)).squeeze(-1)  # [B, D]
    e_term = e.unsqueeze(-1) * k32.unsqueeze(1)  # [B, D, D]

    d_e_term = g * omega_w_eff  # [B, D, D]
    d_e = (d_e_term * k32.unsqueeze(1)).sum(dim=-1)  # [B, D]
    d_state = g * (1.0 - omega_e_eff) - d_e.unsqueeze(-1) * k32.unsqueeze(
        1
    )  # [B, D, D]
    d_k = (d_e_term * e.unsqueeze(-1)).sum(dim=1)  # [B, D]
    d_v = d_e  # [B, D]

    d_omega_w_eff = (g * e_term).sum(dim=-1)  # [B, D]
    d_omega_w = (d_omega_w_eff * write32).sum(dim=-1, keepdim=True)  # [B, 1]
    d_write = d_omega_w_eff * omega_w32  # [B, D]

    d_omega_e_eff = -(g * state32).sum(dim=-1)  # [B, D]
    d_omega_e = (d_omega_e_eff * erase32).sum(dim=-1, keepdim=True)  # [B, 1]
    d_erase = d_omega_e_eff * omega_e32  # [B, D]

    return (
        d_state.to(state.dtype),
        d_k.to(k.dtype),
        d_v.to(v.dtype),
        d_omega_w.to(omega_w.dtype),
        d_omega_e.to(omega_e.dtype),
        d_write.to(write.dtype),
        d_erase.to(erase.dtype),
    )


delta_core_step_op.register_autograd(
    _delta_core_step_out_backward,
    setup_context=_delta_core_step_out_setup,
)
