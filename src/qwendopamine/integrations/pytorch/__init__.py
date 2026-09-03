"""PyTorch-side custom operator registrations.

The public ops in :mod:`qwendopamine.ops` are exposed to PyTorch as
``torch.library.custom_op`` operators. The real implementation lives
in the Taichi kernels (or, when Taichi is unavailable, the
pure-PyTorch reference); the fake/meta implementation lets
``torch.compile`` and the Dynamo/AOT pipeline reason about shapes and
dtypes without executing the kernel; the autograd Function records
the backward rule.

The :mod:`qwendopamine.integrations.pytorch.devices` module provides
the cross-hardware device-detection helper used by the per-device
custom-op kernels: it picks CUDA → XPU → MPS → Vulkan (via Taichi)
→ CPU and migrates tensors accordingly.

Schema:

    qwendopamine::chunk_gdn2                    chunkwise Gated DeltaNet-2 forward (Tensor)
    qwendopamine::chunk_gdn2_with_state         ...returns Tensor[] of [out, state]
    qwendopamine::recurrent_gdn2                single-token Gated DeltaNet-2 forward (Tensor)
    qwendopamine::recurrent_gdn2_with_state     ...returns Tensor[] of [out, state]
    qwendopamine::delta_core_step               Reinforced Delta per-token update (Tensor)

Each registered op is the canonical entry point for code that wants
``torch.compile`` compatibility, ``opcheck`` validation, or PyTorch
shape inference. The :mod:`qwendopamine.ops` package exposes plain
Python functions that delegate to these custom ops when Taichi is
unavailable (and to the Taichi kernel directly when it is — the
custom-op registration is opt-in for callers that need it).
"""

from __future__ import annotations

# Import the modules directly to avoid importing the heavy
# ``qwendopamine.integrations`` parent package (which transitively
# loads HuggingFace transformers and torchvision just to get the
# device-detection helper).
from qwendopamine.integrations.pytorch import autograd, custom_ops, devices

is_autograd_registered = autograd.is_autograd_registered
register_all_autograd = autograd.register_all_autograd

chunk_gdn2_op = custom_ops.chunk_gdn2_op
chunk_gdn2_with_state_op = custom_ops.chunk_gdn2_with_state_op
delta_core_step_op = custom_ops.delta_core_step_op
is_accelerator_kernels_registered = custom_ops.is_accelerator_kernels_registered
is_registered = custom_ops.is_registered
recurrent_gdn2_op = custom_ops.recurrent_gdn2_op
recurrent_gdn2_with_state_op = custom_ops.recurrent_gdn2_with_state_op
register_accelerator_kernels = custom_ops.register_accelerator_kernels
register_all = custom_ops.register_all

__all__ = [
    "chunk_gdn2_op",
    "chunk_gdn2_with_state_op",
    "delta_core_step_op",
    "devices",
    "is_accelerator_kernels_registered",
    "is_autograd_registered",
    "is_registered",
    "recurrent_gdn2_op",
    "recurrent_gdn2_with_state_op",
    "register_accelerator_kernels",
    "register_all",
    "register_all_autograd",
]
