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

GDN2GPT = SurpriseGPT
GDN2GPTConfig = SurpriseGPTConfig

__all__ = [
    "GDN2GPT",
    "Block",
    "CausalSelfAttention",
    "GDN2GPTConfig",
    "LLaMAMLP",
    "RMSNorm",
    "SurpriseGPT",
    "SurpriseGPTConfig",
    "SwiGLU",
    "apply_rotary_emb",
    "build_rope_cache",
    "compute_model_params",
]
