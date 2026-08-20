r"""Reference Qwen3.5 Gated DeltaNet and GDN-2 block re-exports."""

from __future__ import annotations

from qwendopamine.models.gdn2.gdn2 import GatedDeltaNet2
from qwendopamine.models.qwen35.modular_qwen3_5 import Qwen3_5GatedDeltaNet

GatedDeltaNetBlock = Qwen3_5GatedDeltaNet
GatedDeltaNet2Block = GatedDeltaNet2

__all__ = [
    "GatedDeltaNet2",
    "GatedDeltaNet2Block",
    "GatedDeltaNetBlock",
    "Qwen3_5GatedDeltaNet",
]
