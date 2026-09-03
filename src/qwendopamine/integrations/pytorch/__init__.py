"""PyTorch-side custom operator registrations.

The public ops in :mod:`qwendopamine.ops` are exposed to PyTorch as
``torch.library.custom_op`` operators. The real implementation lives
in the Taichi kernels (or, when Taichi is unavailable, the
pure-PyTorch reference); the fake/meta implementation lets
``torch.compile`` and the Dynamo/AOT pipeline reason about shapes and
dtypes without executing the kernel; the autograd Function records
the backward rule.

Schema:

    qwendopamine::chunk_gdn2         chunkwise Gated DeltaNet-2 forward
    qwendopamine::recurrent_gdn2      single-token Gated DeltaNet-2 forward
    qwendopamine::delta_core_step     Reinforced Delta per-token update

Each registered op is the canonical entry point for code that wants
``torch.compile`` compatibility, ``opcheck`` validation, or PyTorch
shape inference. The :mod:`qwendopamine.ops` package exposes plain
Python functions that delegate to these custom ops when Taichi is
unavailable (and to the Taichi kernel directly when it is — the
custom-op registration is opt-in for callers that need it).
"""

from __future__ import annotations

from qwendopamine.integrations.pytorch.autograd import (
    is_autograd_registered,
    register_all_autograd,
)
from qwendopamine.integrations.pytorch.custom_ops import (
    chunk_gdn2_op,
    delta_core_step_op,
    is_registered,
    recurrent_gdn2_op,
    register_all,
)

__all__ = [
    "chunk_gdn2_op",
    "delta_core_step_op",
    "is_autograd_registered",
    "is_registered",
    "recurrent_gdn2_op",
    "register_all",
    "register_all_autograd",
]
