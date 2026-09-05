"""Backend-name normalisation for GDN-2 dispatch.

Moved from ``block.py`` for size.
"""

from __future__ import annotations


def _normalise_backend(name: str) -> str:
    r"""_normalise_backend(name: str) -> str

    Map user-friendly backend names to canonical GDN-2 dispatch IDs.

    Args:
        name (str): Public backend identifier.

    Returns:
        str: Canonical backend identifier used by the dispatch layer.
    """
    if name == "torch":
        return "torch-chunk"
    if name == "compiled":
        return "torch-chunk"
    if name in ("triton", "fla"):
        return "taichi"
    return name


# Backends constant accessible from block module
_GATED_DELTA_NET_BACKENDS = (
    "auto",
    "taichi",
    "torch",
    "torch-chunk",
    "torch-recurrent",
    "compiled",
    "triton",
    "fla",
)
