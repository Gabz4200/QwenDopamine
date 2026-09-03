"""External integrations.

Subpackages:

    - ``huggingface`` — transformers / HF Hub round-trip
    - ``pytorch`` — ``torch.library.custom_op`` registration for the
      public ops, autograd rules, and meta/fake implementations
    - ``safetensors`` — model weight save/load via safetensors
    - ``gguf`` — GGUF weight interoperability
    - ``tokenizer`` — tokenizer / sentencepiece adapters
"""

from __future__ import annotations

from qwendopamine.integrations.huggingface import (
    GDN2HFBlock,
    GDN2HFConfig,
    HFIntegration,
)
from qwendopamine.integrations.safetensors import (
    load_safetensors,
    save_safetensors,
)

__all__ = [
    "GDN2HFBlock",
    "GDN2HFConfig",
    "HFIntegration",
    "load_safetensors",
    "save_safetensors",
]
