"""Shared backend-selection registry for ops.

Backends are discovered through :func:`register_backend` and resolved by
name in :func:`resolve_backend`. Registration is idempotent; repeated
``register_backend`` calls for the same name replace the previous factory.
The default backends are registered on first import so existing call sites
keep working without change.
"""

from __future__ import annotations

from collections.abc import Callable


class BackendResolutionError(RuntimeError):
    """Raised when a requested backend cannot be resolved."""


BackendFactory = Callable[[], str]
_registry: dict[str, BackendFactory] = {}
_loaded: dict[str, str] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a backend provider under *name*.

    *factory* is called lazily on first resolution of *name* and must
    return a backend name string.
    """
    _registry[name] = factory
    _loaded.pop(name, None)


def resolve_backend(name: str) -> str:
    """Return the backend name for *name*.

    The provider factory is invoked at most once per registered name.
    Re-registering the same name after resolution invalidates the cache.
    """
    if name not in _registry:
        raise BackendResolutionError(
            f"Backend {name!r} is not registered. "
            "Register it with `register_backend(...)`."
        )
    if name not in _loaded:
        _loaded[name] = _registry[name]()
    return _loaded[name]


def available_backends() -> list[str]:
    """Return the names of all registered backends."""
    return list(_registry)
