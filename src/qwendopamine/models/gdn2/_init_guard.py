"""Init-time guard shared by GDN-2 initialisation hooks.

A small helper for the "initialise-once per module instance" pattern
used by ``GatedDeltaNet2._initialize_weights`` and ``GDN2Host._init_weights``.
Both functions guard against re-initialising parameters that HF has
already populated (e.g. when loading a state dict triggers a second
``apply`` pass).
"""

from __future__ import annotations

from torch import nn

# Module-level attribute name used to mark a module as "already
# initialised". Avoids the ``cast(Any, module)._is_hf_initialized = True``
# pattern that litters the initialisers.
_INITIALISED_ATTR = "_gdn2_initialised"


def is_already_initialised(module: nn.Module) -> bool:
    """Return True if ``module`` was already initialised by a GDN-2 hook."""
    return bool(getattr(module, _INITIALISED_ATTR, False))


def mark_initialised(module: nn.Module) -> None:
    """Flag ``module`` as initialised so subsequent hooks short-circuit."""
    module._gdn2_initialised = True  # type: ignore[attr-defined]
    # Keep the legacy attribute name in sync so external code that
    # inspects the old key still works.
    module._is_hf_initialized = True  # type: ignore[attr-defined]


__all__ = [
    "_INITIALISED_ATTR",
    "is_already_initialised",
    "mark_initialised",
]
