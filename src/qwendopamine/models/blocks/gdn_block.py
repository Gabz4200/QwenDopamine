"""Reference Qwen3.5 Gated DeltaNet, GDN-2 & GatedSurpriseNet blocks."""

from __future__ import annotations

from qwendopamine.models.gated_surprise_net import GatedSurpriseNetAdam
from qwendopamine.models.gdn2.gdn2 import GatedDeltaNet2
from qwendopamine.models.qwen35.modular_qwen3_5 import Qwen3_5GatedDeltaNet

# Aliases for GDN-1, GDN-2 and GatedSurpriseNet blocks
GatedDeltaNetBlock = Qwen3_5GatedDeltaNet
GatedDeltaNet2Block = GatedDeltaNet2
GatedSurpriseNetBlock = GatedSurpriseNetAdam

__all__ = [
    "GatedDeltaNet2",
    "GatedDeltaNet2Block",
    "GatedDeltaNetBlock",
    "GatedSurpriseNetAdam",
    "GatedSurpriseNetBlock",
    "Qwen3_5GatedDeltaNet",
]
