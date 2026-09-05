"""``torch.library.custom_op`` registrations for the public ops.

Re-exports all public operators and helpers from the split submodules
so that existing imports continue to work unchanged.

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

Backend selection
-----------------
Taichi handles backend selection itself. The kernel runtime
picks CUDA → Vulkan → Metal/OpenGL → CPU on its own; the
ops below do **not** select a backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
else:
    pass  # runtime import for decorators


__all__ = [
    "_chunk_gdn2_fake",
    "_chunk_gdn2_with_state_fake",
    "_delta_core_step_fake",
    "_recurrent_gdn2_fake",
    "_recurrent_gdn2_with_state_fake",
    "chunk_gdn2_op",
    "chunk_gdn2_with_state_op",
    "delta_core_step_op",
    "is_registered",
    "recurrent_gdn2_op",
    "recurrent_gdn2_with_state_op",
    "register_accelerator_kernels",
    "register_all",
]

# Import the public ops and fake implementations from the split submodules
# so that ``from qwendopamine.integrations.pytorch.custom_ops import ...``
# continues to work identically to the original module.
from .chunk import (
    _chunk_gdn2_fake,
    _chunk_gdn2_with_state_fake,
    chunk_gdn2_op,
    chunk_gdn2_with_state_op,
)
from .delta import (
    _delta_core_step_fake,
    delta_core_step_op,
)
from .recurrent import (
    _recurrent_gdn2_fake,
    _recurrent_gdn2_with_state_fake,
    recurrent_gdn2_op,
    recurrent_gdn2_with_state_op,
)


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
    from .register import register_accelerator_kernels as _ra

    _ra()


_ACCEL_REGISTERED: bool = False


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


_REGISTERED: bool = False


def is_registered() -> bool:
    """Return True if the public ops are registered as ``torch.ops``."""
    return _REGISTERED


# Eagerly register at import time so the ``torch.ops.qwendopamine.*``
# namespace is available as soon as a caller imports this module.
register_all()
