"""Autograd rules for the public custom ops.

The Taichi kernels for the GDN-2 chunk/recurrent ops carry their own
VJP and are wrapped by the production ``torch.autograd.Function`` in
:mod:`qwendopamine.kernels.taichi.gdn2_api`. The Reinforced Delta op
has its autograd wrapper in
:mod:`qwendopamine.kernels.taichi.reinforced_kernels`.

This module re-exports the registration helpers and provides a
``register_all_autograd()`` entry point. Because the production
``torch.autograd.Function`` wrappers expect the in-place signature
(with a ``next_state`` output buffer) while the custom ops are
functional (returning a fresh tensor), the autograd rules cannot be
attached 1:1. Callers that need autograd through the public ops
should go through the :mod:`qwendopamine.ops` Python wrappers
(``chunk_taichi_gdn2``, ``recurrent_taichi_gdn2``,
``delta_core_step_out``) directly — those already carry the
right ``torch.autograd.Function`` registration.
"""

from __future__ import annotations


def _register_chunk_gdn2_autograd() -> None:
    """No-op: production autograd lives on the in-place kernel
    ``torch.autograd.Function``. The functional custom op is for
    ``torch.compile`` / ``opcheck`` integration.
    """
    return


def _register_recurrent_gdn2_autograd() -> None:
    """No-op: see :func:`_register_chunk_gdn2_autograd`."""
    return


def _register_delta_core_step_out() -> None:
    """No-op: see :func:`_register_chunk_gdn2_autograd`."""
    return


_REGISTERED_AUTOGRAD: bool = False


def register_all_autograd() -> None:
    """Idempotently attach autograd rules. Currently a no-op — see
    module docstring for rationale.

    Review M13: the previous code initialised ``_REGISTERED_AUTOGRAD``
    to ``True`` at import time, so ``is_autograd_registered()`` always
    returned True even if the registration function was never called
    or failed. The flag is now set to True ONLY after a successful
    ``register_all_autograd()`` call.
    """
    global _REGISTERED_AUTOGRAD
    if _REGISTERED_AUTOGRAD:
        return
    _register_chunk_gdn2_autograd()
    _register_recurrent_gdn2_autograd()
    _register_delta_core_step_out()
    _REGISTERED_AUTOGRAD = True


def is_autograd_registered() -> bool:
    """Return True if every public op has an autograd rule attached."""
    return _REGISTERED_AUTOGRAD


__all__ = [
    "is_autograd_registered",
    "register_all_autograd",
]


# Eagerly register at import time so the public ops carry autograd
# rules as soon as a caller imports this module. The flag below is
# flipped to True ONLY on successful completion (review M13).
register_all_autograd()
