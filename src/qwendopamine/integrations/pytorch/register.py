"""Device routing and accelerator kernel registration utilities.

Provides :func:`_route_to_active_device` for migrating inputs to the
active Taichi backend and :func:`register_accelerator_kernels` for
registering per-accelerator kernels on the public custom ops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from qwendopamine.integrations.pytorch import devices as _devices

if TYPE_CHECKING:
    from torch._library.custom_ops import CustomOpDef
else:
    CustomOpDef = object  # type: ignore[assignment,misc]


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


_ACCEL_REGISTERED: bool = False


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
        r"""(args: object) -> object - Per-device kernel wrapper.

        Migrate input tensors to active accelerator device before calling
        the op body. See :func:`default_device` for device selection.

        Args:
            *args (object): Positional args passed to the op body.
        Returns:
            object: Op result from ``body(*migrated_args)``.
        """
        moved = list(args)
        for i in tensor_arg_indices:
            t = moved[i]
            if isinstance(t, torch.Tensor) and t.device != active:
                moved[i] = t.to(active)
        return body(*moved)

    op.register_kernel(device_name)(kernel)


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

    # Lazy import to avoid circular imports at module load time.
    from .chunk import (
        _chunk_gdn2_body,
        _chunk_gdn2_with_state_body,
        chunk_gdn2_op,
        chunk_gdn2_with_state_op,
    )
    from .delta import _delta_core_step_body, delta_core_step_op
    from .recurrent import (
        _recurrent_gdn2_body,
        _recurrent_gdn2_with_state_body,
        recurrent_gdn2_op,
        recurrent_gdn2_with_state_op,
    )

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
