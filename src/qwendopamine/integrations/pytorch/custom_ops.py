"""``torch.library.custom_op`` registrations for the public ops.

The skill :mod:`taichi-pytorch-interop` recommends registering custom
ops via ``torch.library.custom_op`` so they are ``torch.compile``-,
``opcheck``-, and ``FakeTensor``-compatible. Each op here exposes:

    - the real (eager) implementation, which delegates to the Taichi
      kernel when available and to the pure-PyTorch reference
      otherwise;
    - a fake/meta implementation that returns the right shape and dtype
      without touching the kernel, so ``torch.compile`` can trace the
      graph;
    - the autograd rule is the existing ``torch.autograd.Function`` in
      :mod:`qwendopamine.kernels.taichi.reinforced_kernels` for the
      Reinforced Delta op; the GDN-2 ops are not autograd-differentiable
      on their own (the chunk/recurrent kernels carry the VJP in the
      production path).

Custom-op registration is **opt-in** — callers call
:func:`register_all` once at process startup, then route their model
code through the registered ops via ``torch.ops.qwendopamine.chunk_gdn2``
instead of the plain Python wrapper in :mod:`qwendopamine.ops`.

Schemas (the op return type is constrained to ``Tensor[]`` by
``torch.library.custom_op``; the GDN-2 ops therefore return
``[output, final_state_or_none]`` as a list):

    qwendopamine::chunk_gdn2(q, k, v, g, b, w, initial_state, output_final_state)
        -> Tensor[]   [output, final_state?]
    qwendopamine::recurrent_gdn2(q, k, v, g, b, w, initial_state, output_final_state)
        -> Tensor[]   [output, final_state?]
    qwendopamine::delta_core_step(state, k, v, omega_w, omega_e, write, erase, next_state)
        -> Tensor     next_state
"""

from __future__ import annotations

import torch
from torch.library import custom_op

# Public ops the registrations delegate to. These are imported lazily
# inside each op body so the registration module itself does not
# trigger Taichi initialisation at import time (per the
# taichi-pytorch-interop skill: "Avoid calling ti.init() at module
# import time").


# ---------------------------------------------------------------------------
# GDN-2 chunk
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
    output_final_state: bool = False,
) -> list[torch.Tensor]:
    """Public GDN-2 chunkwise forward registered as a ``torch.library`` op.

    Returns ``[output, final_state]`` as a list because
    ``torch.library.custom_op`` only supports ``Tensor`` /
    ``Tensor[]`` return types.
    """
    from qwendopamine.ops import chunk_taichi_gdn2

    out, state = chunk_taichi_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return [out, state] if state is not None else [out]


@chunk_gdn2_op.register_fake
def _chunk_gdn2_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> list[torch.Tensor]:
    """Fake implementation used by ``torch.compile``/``FakeTensor``.

    Strides match the real (eager) Taichi kernel output after the
    contiguous reshape applied at the op boundary.
    """
    B, T, H, _ = q.shape
    V = v.shape[-1]
    out = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    if output_final_state:
        state = torch.empty(B, H, q.shape[-1], V, dtype=q.dtype, device=q.device)
        return [out, state]
    return [out]


# ---------------------------------------------------------------------------
# GDN-2 recurrent
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
    output_final_state: bool = False,
) -> list[torch.Tensor]:
    """Public GDN-2 single-token recurrent forward registered as a ``torch.library`` op."""
    from qwendopamine.ops import recurrent_taichi_gdn2

    out, state = recurrent_taichi_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return [out, state] if state is not None else [out]


@recurrent_gdn2_op.register_fake
def _recurrent_gdn2_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> list[torch.Tensor]:
    """Fake implementation used by ``torch.compile``/``FakeTensor``."""
    B, _, H, _ = q.shape
    V = v.shape[-1]
    out = torch.empty(B, q.shape[1], H, V, dtype=q.dtype, device=q.device)
    if output_final_state:
        state = torch.empty(B, H, q.shape[-1], V, dtype=q.dtype, device=q.device)
        return [out, state]
    return [out]


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
    """Reinforced Delta per-token update registered as a ``torch.library`` op.

    Functional: returns a fresh ``next_state`` tensor. The
    in-place variant lives at
    :mod:`qwendopamine.ops.reward.delta_core_step_autograd` for
    callers that want to reuse an output buffer; this functional
    version exists for ``torch.compile``/``register_autograd``
    compatibility.
    """
    from qwendopamine.ops.reward import _reward_torch_step

    return _reward_torch_step(state, k, v, omega_w, omega_e, write, erase)


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
    """Fake implementation. Same shape as ``state``."""
    return torch.empty_like(state)


# ---------------------------------------------------------------------------
# Registration entry point
# ---------------------------------------------------------------------------
_REGISTERED: bool = False


def register_all() -> None:
    """Idempotently register every op in this module with ``torch.ops``.

    Safe to call multiple times. The first call sets up the
    ``qwendopamine`` namespace in ``torch.ops``; subsequent calls are
    no-ops.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    # ``@custom_op`` already registers the op eagerly when the module
    # is imported. This function exists so callers have a single
    # entry point and can detect whether registration has happened.
    _REGISTERED = True


def is_registered() -> bool:
    """Return True if the public ops are registered as ``torch.ops``."""
    return _REGISTERED


# Eagerly register at import time so the ``torch.ops.qwendopamine.*``
# namespace is available as soon as a caller imports this module.
register_all()


__all__ = [
    "chunk_gdn2_op",
    "delta_core_step_op",
    "is_registered",
    "recurrent_gdn2_op",
    "register_all",
]
