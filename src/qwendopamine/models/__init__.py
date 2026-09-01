"""Qwen3.5 research model layer.

The model packages under ``qwen35`` and ``infinidopamine`` import
``transformers.models.qwen3_next`` and ``qwen3_vl`` at module top level, which
adds ~12s of import cost on cold start (measured on this machine, transformers
5.15.0). To keep ``import qwendopamine.models`` cheap, model classes are
deferred until first attribute access via PEP 562 ``__getattr__``. Touching a
model class triggers the underlying submodule import; subsequent accesses are
O(1) because the loaded module is cached in ``sys.modules``.

Measured (this machine): ``import qwendopamine.models`` ≈ 0.0s with the lazy
registry, ≈ 18s without it. The light core primitives stay eagerly available.
"""

from __future__ import annotations

from typing import Any

from qwendopamine.models.core.embeddings import PositionEmbeddings, TokenEmbeddings
from qwendopamine.models.core.normalization import RMSNorm
from qwendopamine.models.core.output_head import LMHead
from qwendopamine.models.model_factory import (
    ResearchDecoder,
    build_model,
    build_reference_model,
)

__all__ = [
    "LMHead",
    "PositionEmbeddings",
    "RMSNorm",
    "ResearchDecoder",
    "TokenEmbeddings",
    "build_model",
    "build_reference_model",
]

_MODULE_ATTRS: dict[str, str] = {
    "GDN2GPT": "qwendopamine.models.gdn2_gpt",
    "GDN2GPTConfig": "qwendopamine.models.gdn2_gpt",
    "GatedDeltaNet2": "qwendopamine.models.gdn2",
    "InfiniDopamineConfig": "qwendopamine.models.infinidopamine",
    "InfiniDopamineDecoderLayer": "qwendopamine.models.infinidopamine",
    "InfiniDopamineForCausalLM": "qwendopamine.models.infinidopamine",
    "InfiniDopamineGatedDeltaNet": "qwendopamine.models.infinidopamine",
    "InfiniDopamineGatedRewardNet": "qwendopamine.models.infinidopamine",
    "InfiniDopamineModel": "qwendopamine.models.infinidopamine",
    "InfiniDopaminePreTrainedModel": "qwendopamine.models.infinidopamine",
    "InfiniDopamineTextConfig": "qwendopamine.models.infinidopamine",
    "InfiniDopamineTextModel": "qwendopamine.models.infinidopamine",
    "Qwen3_5Config": "qwendopamine.models.qwen35",
    "Qwen3_5DecoderLayer": "qwendopamine.models.qwen35",
    "Qwen3_5ForCausalLM": "qwendopamine.models.qwen35",
    "Qwen3_5GatedDeltaNet": "qwendopamine.models.qwen35",
    "Qwen3_5Model": "qwendopamine.models.qwen35",
    "Qwen3_5PreTrainedModel": "qwendopamine.models.qwen35",
    "Qwen3_5TextConfig": "qwendopamine.models.qwen35",
    "Qwen3_5TextModel": "qwendopamine.models.qwen35",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_ATTRS:
        import importlib

        module = importlib.import_module(_MODULE_ATTRS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
