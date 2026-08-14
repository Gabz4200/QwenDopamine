"""Reference Qwen3.5 Gated DeltaNet block."""

from __future__ import annotations

from qwendopamine.models.qwen35.modular_qwen3_5 import Qwen3_5GatedDeltaNet

# Backwards compatibility alias
GatedDeltaNetBlock = Qwen3_5GatedDeltaNet
GatedDeltaNet2Block = Qwen3_5GatedDeltaNet

__all__ = [
    "GatedDeltaNet2Block",
    "GatedDeltaNetBlock",
    "Qwen3_5GatedDeltaNet",
]
