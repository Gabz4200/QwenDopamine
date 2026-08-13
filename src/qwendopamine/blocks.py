"""Top-level re-export for blocks subpackage.

Allows ``from qwendopamine.blocks import GDN2Mixer`` instead of
``from qwendopamine.models.blocks import GDN2Mixer``.
"""

from __future__ import annotations

from qwendopamine.models.blocks import (
    GDN2Mixer,
    GDN2Projections,
    GatedDeltaNet2Block,
    dispatch_gdn2,
)

__all__ = [
    "GDN2Mixer",
    "GDN2Projections",
    "GatedDeltaNet2Block",
    "dispatch_gdn2",
]
