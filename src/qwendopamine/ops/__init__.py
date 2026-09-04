"""Public operations layer.

This package exposes the stable, model-facing operations that the rest of
the codebase consumes. The dependency direction is:

    models/* ───> ops/* ───> kernels/taichi/*
                  └──> models/gdn2/recurrence/*  (pure-PyTorch reference)
                  └──> models/reinforced/canonical_reference.py

Models MUST import operations from this package — never directly from
``qwendopamine.kernels.taichi.*``. Kernels are an implementation
detail; ops are the public contract.

Backend dispatch: Taichi handles backend selection itself. The kernel
runtime is initialised on first use via
:func:`qwendopamine.kernels.taichi.runtime._initialise` and Taichi picks
CUDA → Vulkan → Metal/OpenGL → CPU. The pure-PyTorch reference is the
fallback when Taichi is unavailable (e.g. ``taichi`` not installed).
"""

from __future__ import annotations

from qwendopamine.ops.gdn2 import chunk_taichi_gdn2, recurrent_taichi_gdn2
from qwendopamine.ops.reward import delta_core_step, delta_core_step_out

__all__ = [
    "chunk_taichi_gdn2",
    "delta_core_step",
    "delta_core_step_out",
    "recurrent_taichi_gdn2",
]
