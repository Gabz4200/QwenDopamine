"""Hybrid GPT Decoder Architecture with central GatedSurpriseNet token mixer."""

from qwendopamine.models.surprise_gpt.config import SurpriseGPTConfig
from qwendopamine.models.surprise_gpt.model import (
    Block,
    CausalSelfAttention,
    LLaMAMLP,
    RMSNorm,
    SurpriseGPT,
    SwiGLU,
    apply_rotary_emb,
    build_rope_cache,
    compute_model_params,
)

__all__ = [
    "Block",
    "CausalSelfAttention",
    "LLaMAMLP",
    "RMSNorm",
    "SurpriseGPT",
    "SurpriseGPTConfig",
    "SwiGLU",
    "apply_rotary_emb",
    "build_rope_cache",
    "compute_model_params",
]
