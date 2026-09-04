"""Hugging Face integration for QwenDopamine.

Split into focused modules:

- :mod:`qwendopamine.integrations.huggingface.configs` — HF PreTrainedConfig
  adapters and optional-import fallbacks for ``transformers`` symbols.
- :mod:`qwendopamine.integrations.huggingface.block` — :class:`GDN2HFBlock`
  nn.Module wrapper around :class:`GatedDeltaNet2`.
- :mod:`qwendopamine.integrations.huggingface.integration` —
  :class:`HFIntegration` facade with registration, load, save, and
  quantization entry points.

The flat public re-exports below keep ``from qwendopamine.integrations.huggingface
import HFIntegration`` working for every historical caller.
"""

from qwendopamine.integrations.huggingface.block import GDN2HFBlock
from qwendopamine.integrations.huggingface.configs import (
    GDN2HFConfig,
    InfiniDopamineGDN2HFConfig,
    PreTrainedConfig,
    Qwen35GDN2HFConfig,
)
from qwendopamine.integrations.huggingface.integration import HFIntegration

__all__ = [
    "GDN2HFBlock",
    "GDN2HFConfig",
    "HFIntegration",
    "InfiniDopamineGDN2HFConfig",
    "PreTrainedConfig",
    "Qwen35GDN2HFConfig",
]
