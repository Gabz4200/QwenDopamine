# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Backend resolution and execution dispatch for GDN-2.

The Taichi backend is the single hardware-accelerated engine; Taichi
itself picks CUDA → Vulkan → Metal/OpenGL → CPU. Backend resolution
fails fast: if Taichi is unavailable, an error is raised.
"""

from __future__ import annotations

from qwendopamine.ops._backend_registry import resolve_backend

GDN2_BACKENDS = (
    "auto",
    "taichi",
    "torch",
    "torch-chunk",
    "torch-recurrent",
    "compiled",
    "triton",
    "fla",
)


def resolve_gdn2_backend(
    requested: str,
    *,
    training: bool,
    seq_len: int,
) -> str:
    r"""Resolve the concrete GDN-2 execution backend for a forward call.

    The Taichi backend is the single hardware-accelerated engine; Taichi
    itself picks CUDA → Vulkan → Metal/OpenGL → CPU. ``"auto"`` selects
    Taichi when available. If Taichi is unavailable, this raises
    ``RuntimeError``.

    Args:
        requested: The backend name requested by the caller (``"auto"``,
            ``"taichi"``, ``"torch"``, ``"torch-chunk"``,
            ``"torch-recurrent"``, ``"compiled"``, ``"triton"``, ``"fla"``).
        training: Whether the model is in training mode.
        seq_len: Current sequence length (used for heuristic defaults).

    Returns:
        The resolved backend name string.

    Raises:
        ValueError: If *requested* is not a valid backend name.
        RuntimeError: If Taichi is required but unavailable.
    """
    if requested not in GDN2_BACKENDS:
        raise ValueError(
            f"Invalid GDN-2 backend '{requested}'. Valid backends: {list(GDN2_BACKENDS)}"
        )
    if requested != "auto":
        # The CUDA-bound triton/fla paths were replaced by Taichi; the
        # old scalar names now route to the equivalent path.
        if requested in ("triton", "fla"):
            return "taichi"
        if requested == "compiled":
            return "torch-chunk"
        return resolve_backend(requested)

    # "auto" — Taichi is a hard dependency, always available.
    return "taichi"


__all__ = [
    "GDN2_BACKENDS",
    "resolve_gdn2_backend",
]
