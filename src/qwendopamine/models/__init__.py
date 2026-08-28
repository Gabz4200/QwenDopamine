"""Qwen3.5 research model layer.

Public symbols are re-exported here for convenience, but heavy model imports
are deferred until first access to avoid forcing the full model tree on every
``import qwendopamine.models``.
"""

from __future__ import annotations

from typing import Any

from qwendopamine.models.embeddings import PositionEmbeddings, TokenEmbeddings
from qwendopamine.models.model_factory import (
    ResearchDecoder,
    build_model,
    build_reference_model,
)
from qwendopamine.models.normalization import RMSNorm
from qwendopamine.models.output_head import LMHead

# Lightweight imports available immediately.
__all__ = [
    "LMHead",
    "PositionEmbeddings",
    "RMSNorm",
    "ResearchDecoder",
    "TokenEmbeddings",
    "build_model",
    "build_reference_model",
]

# Heavy model imports are deferred to reduce import-time fan-out.
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
