"""GDN-2 GPT Decoder Architecture (lit_gpt-inspired hybrid transformer with GatedDeltaNet2)."""

from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig
from qwendopamine.models.gdn2_gpt.model import (
    GDN2GPT,
    Block,
    CausalSelfAttention,
    LLaMAMLP,
    RMSNorm,
    SwiGLU,
    apply_rotary_emb,
    build_rope_cache,
    compute_model_params,
)

__all__ = [
    "GDN2GPT",
    "Block",
    "CausalSelfAttention",
    "GDN2GPTConfig",
    "LLaMAMLP",
    "RMSNorm",
    "SwiGLU",
    "apply_rotary_emb",
    "build_rope_cache",
    "compute_model_params",
]
