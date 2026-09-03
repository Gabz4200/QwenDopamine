"""``torch.library.custom_op`` registrations for the public ops.

Implements the operator-oriented modular architecture from
:mod:`taichi-pytorch-interop`: model code calls
``torch.ops.qwendopamine.*`` and PyTorch's subsystems
(``torch.compile``, ``opcheck``, ``FakeTensor``, ``register_autograd``)
treat each op as an opaque boundary.

Schema contract
---------------
Each op below is **functional** (mutates no inputs) and follows the
"stable schema" rule from the
`Custom Python Operators tutorial
<https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html>`_:

    - the number of return Tensors is fixed per op (no maybe-out);
    - the returned Tensors do not alias any input Tensor;
    - the fake kernel matches the real kernel's output metadata
      (shape, dtype, device, layout, strides, storage offset);
    - the fake kernel may inspect metadata but must not read data.

The GDN-2 ops are split into two variants each so the return list
length is **stable** per op:

    ``qwendopamine::chunk_gdn2``        -> ``Tensor``           (output only)
    ``qwendopamine::chunk_gdn2_with_state``
        -> ``Tensor[2]``                  (output, final_state)
    ``qwendopamine::recurrent_gdn2``    -> ``Tensor``           (output only)
    ``qwendopamine::recurrent_gdn2_with_state``
        -> ``Tensor[2]``                  (output, final_state)
    ``qwendopamine::delta_core_step``   -> ``Tensor``           (next_state, fresh)

Device support
--------------
Each op is registered for **CPU and every available accelerator**
(CUDA, XPU, MPS, Vulkan-via-Taichi) via
``CustomOpDef.register_kernel``. When the caller passes a tensor on
a device that does not match the active Taichi arch, the per-device
kernel migrates the inputs to :func:`devices.default_device` first
and migrates the result back to the caller's device on return.
This is the "Taichi picks the backend, but we don't force a
particular device onto callers" policy.

The active Taichi arch is the single source of truth for the
target device; PyTorch's CUDA/XPU/MPS probes determine which
``device_types`` string is registered. On a system where the
Taichi runtime lands on Vulkan, the corresponding kernel routes
through the Taichi CPU path (the Taichi kernel itself copies
CPU tensors to the active arch internally).

Backend selection
-----------------
Taichi handles backend selection itself. The kernel runtime
picks CUDA → Vulkan → Metal/OpenGL → CPU on its own; the
ops below do **not** select a backend.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from torch.library import custom_op

from qwendopamine.integrations.pytorch import devices as _devices

if TYPE_CHECKING:
    from torch._library.custom_ops import CustomOpDef
else:
    CustomOpDef = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Op-body helpers
# ---------------------------------------------------------------------------
# Each op has a private body that takes already-migrated tensors and
# returns the result on the same device. The public custom-op body
# and every per-device kernel route through the body.
def _chunk_gdn2_body(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    from qwendopamine.ops import chunk_taichi_gdn2

    out, _state = chunk_taichi_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=False,
    )
    return out.contiguous()


def _chunk_gdn2_with_state_body(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> list[torch.Tensor]:
    from qwendopamine.ops import chunk_taichi_gdn2

    out, state = chunk_taichi_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=True,
    )
    state_safe = state.contiguous() if state is not None else q.new_empty(0)
    return [out.contiguous(), state_safe]


def _recurrent_gdn2_body(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    from qwendopamine.ops import recurrent_taichi_gdn2

    out, _state = recurrent_taichi_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=False,
    )
    return out.contiguous()


def _recurrent_gdn2_with_state_body(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> list[torch.Tensor]:
    from qwendopamine.ops import recurrent_taichi_gdn2

    out, state = recurrent_taichi_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=True,
    )
    state_safe = state.contiguous() if state is not None else q.new_empty(0)
    return [out.contiguous(), state_safe]


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


# ---------------------------------------------------------------------------
# Per-device dispatch helpers
# ---------------------------------------------------------------------------
# A "migrated" kernel takes the original tensors from the caller,
# moves them to the active device if needed, runs the op body, and
# moves the result back to the caller's device. This is the policy
# described in the module docstring.
_CALLER_DEVICE = "__caller_device__"


def _caller_device_kwargs(
    tensors: Sequence[torch.Tensor | None],
) -> dict[str, torch.device]:
    """Capture the caller's device so we can return results there.

    ``None`` entries (e.g. ``initial_state=None``) are skipped.
    """
    devices: dict[str, torch.device] = {}
    for i, t in enumerate(tensors):
        if t is not None and isinstance(t, torch.Tensor):
            devices[f"{_CALLER_DEVICE}{i}"] = t.device
    return devices


def _restore_device(
    out: torch.Tensor | list[torch.Tensor],
    caller_devices: dict[str, torch.device],
) -> torch.Tensor | list[torch.Tensor]:
    """Move ``out`` back to the caller's device.

    Works for a single Tensor, a list[Tensor], or a list[Tensor, Tensor].
    """
    if not caller_devices:
        return out
    target = next(iter(caller_devices.values()))
    if isinstance(out, list):
        return [t.to(target) if isinstance(t, torch.Tensor) else t for t in out]
    if isinstance(out, torch.Tensor):
        return out.to(target)
    return out


def _route_to_active_device(
    body: Callable[..., object],
    *args: object,
) -> object:
    """Run ``body(*args)`` on the active device and return on the caller's.

    The active device is :func:`devices.default_device` (CPU when no
    accelerator is detected, otherwise the first available CUDA/XPU/MPS
    device). ``None`` arguments are passed through untouched; all
    tensor arguments are migrated to the active device before the
    body runs, and the result is migrated back to the caller's
    device.

    This is the single dispatch entry point used by every CPU
    kernel in this module. The per-accelerator kernels (CUDA,
    XPU, MPS) registered via :func:`register_accelerator_kernels`
    skip this helper because PyTorch's dispatcher already routes
    them to the right device.
    """
    active = _devices.default_device()
    # Capture the caller's device from the first tensor argument.
    caller_dev: torch.device | None = None
    moved = list(args)
    for i, a in enumerate(moved):
        if isinstance(a, torch.Tensor) and caller_dev is None:
            caller_dev = a.device
            break
    for i, a in enumerate(moved):
        if isinstance(a, torch.Tensor) and a.device != active:
            moved[i] = a.to(active)
    out = body(*moved)
    if caller_dev is not None and caller_dev != active:
        if isinstance(out, list):
            return [t.to(caller_dev) if isinstance(t, torch.Tensor) else t for t in out]
        if isinstance(out, torch.Tensor):
            return out.to(caller_dev)
    return out


# ---------------------------------------------------------------------------
# GDN-2 chunk (no state return)
# ---------------------------------------------------------------------------
@custom_op(
    "qwendopamine::chunk_gdn2",
    mutates_args=(),
    device_types="cpu",
)
def chunk_gdn2_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """GDN-2 chunkwise forward; returns the output only.

    Functional contract: returns a fresh Tensor. The final state is
    discarded by design — use ``qwendopamine::chunk_gdn2_with_state``
    when the carry matters.

    The CPU kernel migrates inputs to the active device
    (:func:`qwendopamine.integrations.pytorch.devices.default_device`)
    if they are not already on CPU, runs the op, and returns the
    result on the caller's original device. This means callers can
    pass CPU tensors on a system with a GPU available and the op
    will still produce the right answer — Taichi handles the
    CPU↔GPU copy internally.
    """
    result = _route_to_active_device(_chunk_gdn2_body, q, k, v, g, b, w, initial_state)
    assert isinstance(result, torch.Tensor)
    return result


@chunk_gdn2_op.register_fake
def _chunk_gdn2_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fake implementation. Output is a fresh contiguous ``[B, T, H, V]``."""
    return torch.empty(
        q.shape[0],
        q.shape[1],
        q.shape[2],
        v.shape[-1],
        dtype=q.dtype,
        device=q.device,
    )


# ---------------------------------------------------------------------------
# GDN-2 chunk (with final state)
# ---------------------------------------------------------------------------
@custom_op(
    "qwendopamine::chunk_gdn2_with_state",
    mutates_args=(),
    device_types="cpu",
)
def chunk_gdn2_with_state_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """GDN-2 chunkwise forward; returns ``[output, final_state]``.

    The final state is a fresh Tensor (never aliases ``initial_state``).
    Use this variant when the next iteration needs the carry.
    """
    result = _route_to_active_device(
        _chunk_gdn2_with_state_body, q, k, v, g, b, w, initial_state
    )
    assert isinstance(result, list)
    return result


@chunk_gdn2_with_state_op.register_fake
def _chunk_gdn2_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Fake implementation. Returns ``[out, state]`` with correct strides."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    out = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    state = torch.empty(B, H, K, V, dtype=q.dtype, device=q.device)
    return [out, state]


# ---------------------------------------------------------------------------
# GDN-2 recurrent (no state return)
# ---------------------------------------------------------------------------
@custom_op(
    "qwendopamine::recurrent_gdn2",
    mutates_args=(),
    device_types="cpu",
)
def recurrent_gdn2_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """GDN-2 single-token recurrent forward; returns the output only."""
    result = _route_to_active_device(
        _recurrent_gdn2_body, q, k, v, g, b, w, initial_state
    )
    assert isinstance(result, torch.Tensor)
    return result


@recurrent_gdn2_op.register_fake
def _recurrent_gdn2_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fake implementation. Output is a fresh contiguous ``[B, T, H, V]``."""
    return torch.empty(
        q.shape[0],
        q.shape[1],
        q.shape[2],
        v.shape[-1],
        dtype=q.dtype,
        device=q.device,
    )


# ---------------------------------------------------------------------------
# GDN-2 recurrent (with final state)
# ---------------------------------------------------------------------------
@custom_op(
    "qwendopamine::recurrent_gdn2_with_state",
    mutates_args=(),
    device_types="cpu",
)
def recurrent_gdn2_with_state_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """GDN-2 single-token recurrent forward; returns ``[output, final_state]``."""
    result = _route_to_active_device(
        _recurrent_gdn2_with_state_body, q, k, v, g, b, w, initial_state
    )
    assert isinstance(result, list)
    return result


@recurrent_gdn2_with_state_op.register_fake
def _recurrent_gdn2_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Fake implementation. Returns ``[out, state]`` with correct strides."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    out = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    state = torch.empty(B, H, K, V, dtype=q.dtype, device=q.device)
    return [out, state]


# ---------------------------------------------------------------------------
# Reinforced Delta per-step
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Per-accelerator kernel registration
# ---------------------------------------------------------------------------
# Register a per-device kernel for every accelerator the host
# exposes. The kernel migrates the inputs to the active device, runs
# the same body the CPU kernel uses, and returns the result on the
# active device (PyTorch's standard dispatcher returns the result
# on the device the kernel produced it on, so callers see the
# accelerator-native output without a migration).
def register_accelerator_kernels() -> None:
    """Register one kernel per available accelerator device type.

    Idempotent. The migration policy: inputs are moved to
    :func:`devices.default_device` if they are not already there.
    The body returns on the active device.

    This is **not** called at import time — probing the active
    Taichi arch forces a full ``ti.init()`` on the first call, which
    is slow on the first Vulkan launch. Call this after you have
    already invoked some other Taichi op (or after explicitly
    initialising the runtime) so the cost is amortised.
    """
    global _ACCEL_REGISTERED
    if _ACCEL_REGISTERED:
        return
    available = _devices.detect_available_devices()
    # Each entry: (op, body-fn, args-of-the-op)
    per_op_specs: list[tuple[CustomOpDef, Callable[..., object], list[int]]] = [
        (
            chunk_gdn2_op,
            _chunk_gdn2_body,
            [0, 1, 2, 3, 4, 5],  # all tensor args are migrated
        ),
        (
            chunk_gdn2_with_state_op,
            _chunk_gdn2_with_state_body,
            [0, 1, 2, 3, 4, 5],
        ),
        (
            recurrent_gdn2_op,
            _recurrent_gdn2_body,
            [0, 1, 2, 3, 4, 5],
        ),
        (
            recurrent_gdn2_with_state_op,
            _recurrent_gdn2_with_state_body,
            [0, 1, 2, 3, 4, 5],
        ),
        (
            delta_core_step_op,
            _delta_core_step_body,
            [0, 1, 2, 3, 4, 5, 6],  # all 7 args are tensors
        ),
    ]
    for op, body, tensor_arg_indices in per_op_specs:
        for dev_name in available:
            if dev_name == "cpu":
                continue  # CPU kernel is already the default registration
            if dev_name not in {"cuda", "xpu", "mps"}:
                # Vulkan / Metal / OpenGL have no PyTorch device; the
                # active-device routing already covers them via the
                # CPU kernel (which returns on the active device when
                # the active device differs from the caller's CPU).
                continue
            _register_one(op, body, tensor_arg_indices, dev_name)
    _ACCEL_REGISTERED = True


_ACCEL_REGISTERED: bool = False


def is_accelerator_kernels_registered() -> bool:
    """Return True if the per-accelerator kernels have been registered."""
    return _ACCEL_REGISTERED


def _register_one(
    op: CustomOpDef,
    body: Callable[..., object],
    tensor_arg_indices: list[int],
    device_name: str,
) -> None:
    """Register a single ``device_name`` kernel on ``op`` that migrates inputs.

    Captures the caller's device on the way in and returns results
    on the active device. The result is *not* migrated back to the
    caller's device: PyTorch's dispatcher returns what the kernel
    returned, and the standard expectation is "the output is on the
    accelerator you asked for". If the caller wants the result on
    CPU, they can ``.to("cpu")`` it themselves.
    """
    active = _devices.default_device()

    def kernel(*args: object) -> object:
        moved = list(args)
        for i in tensor_arg_indices:
            t = moved[i]
            if isinstance(t, torch.Tensor) and t.device != active:
                moved[i] = t.to(active)
        return body(*moved)

    op.register_kernel(device_name)(kernel)


# ---------------------------------------------------------------------------
# Autograd registration
# ---------------------------------------------------------------------------
# Per the Custom Operators manual, the autograd rule must be registered
# after the base op passes opcheck. We use register_autograd with a
# setup formula that re-runs the per-step PyTorch math (the same path
# the fake kernel would trace) so gradients are correct in the
# torch.compile / AOT autograd paths.
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


# ---------------------------------------------------------------------------
# Registration entry point
# ---------------------------------------------------------------------------
_REGISTERED: bool = False


def register_all() -> None:
    """Idempotently register every op in this module with ``torch.ops``.

    The ``@custom_op`` decorator already registers each op eagerly at
    module import time; this function exists so callers can detect
    whether the registration has happened.

    The per-accelerator kernel registration is a **separate, opt-in
    step** — call :func:`register_accelerator_kernels` after the
    Taichi runtime has been initialised by some other code path.
    Doing it here would force a slow first-init on every import.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True


def is_registered() -> bool:
    """Return True if the public ops are registered as ``torch.ops``."""
    return _REGISTERED


# Eagerly register at import time so the ``torch.ops.qwendopamine.*``
# namespace is available as soon as a caller imports this module.
register_all()


__all__ = [
    "chunk_gdn2_op",
    "chunk_gdn2_with_state_op",
    "delta_core_step_op",
    "is_accelerator_kernels_registered",
    "is_registered",
    "recurrent_gdn2_op",
    "recurrent_gdn2_with_state_op",
    "register_accelerator_kernels",
    "register_all",
]
