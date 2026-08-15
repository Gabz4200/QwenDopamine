"""Qwen3.5 research model layer."""

from qwendopamine.models.embeddings import PositionEmbeddings, TokenEmbeddings
from qwendopamine.models.gdn2 import GatedDeltaNet2
from qwendopamine.models.model_factory import (
    ResearchDecoder,
    build_model,
    build_reference_model,
)
from qwendopamine.models.normalization import RMSNorm
from qwendopamine.models.output_head import LMHead
from qwendopamine.models.qwen35 import (
    Qwen3_5Config,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
)
from qwendopamine.models.surprise_gpt import (
    GDN2GPT,
    GDN2GPTConfig,
    SurpriseGPT,
    SurpriseGPTConfig,
)

__all__ = [
    "GDN2GPT",
    "GDN2GPTConfig",
    "GatedDeltaNet2",
    "LMHead",
    "PositionEmbeddings",
    "Qwen3_5Config",
    "Qwen3_5DecoderLayer",
    "Qwen3_5ForCausalLM",
    "Qwen3_5GatedDeltaNet",
    "Qwen3_5Model",
    "Qwen3_5PreTrainedModel",
    "Qwen3_5TextConfig",
    "Qwen3_5TextModel",
    "RMSNorm",
    "ResearchDecoder",
    "SurpriseGPT",
    "SurpriseGPTConfig",
    "TokenEmbeddings",
    "build_model",
    "build_reference_model",
]
