"""Re-exports for the Taichi kernels layer.

Provides the public API that downstream code may import:

    - ``is_available``, ``taichi_arch``, ``require`` (from ``runtime``)
    - ``chunk_taichi_gdn2``, ``recurrent_taichi_gdn2`` (from ``gdn2_api``)
    - ``delta_core_step_autograd`` (from ``reinforced_kernels``)

The kernel implementations themselves live in their respective modules:
:mod:`qwendopamine.kernels.taichi.runtime`,
:mod:`qwendopamine.kernels.taichi.gdn2_api`,
:mod:`qwendopamine.kernels.taichi.gdn2_kernels`,
:mod:`qwendopamine.kernels.taichi.reinforced_kernels`.

Models MUST NOT import these directly; consume the public ops via
:mod:`qwendopamine.ops` instead.
"""

from __future__ import annotations

from qwendopamine.kernels.taichi.gdn2_api import (
    chunk_taichi_gdn2,
    recurrent_taichi_gdn2,
)
from qwendopamine.kernels.taichi.reinforced_kernels import delta_core_step_autograd
from qwendopamine.kernels.taichi.runtime import is_available, require, taichi_arch

__all__ = [
    "chunk_taichi_gdn2",
    "delta_core_step_autograd",
    "is_available",
    "recurrent_taichi_gdn2",
    "require",
    "taichi_arch",
]
