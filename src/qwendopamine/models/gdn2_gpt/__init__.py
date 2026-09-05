"""GDN-2 GPT Decoder Architecture (lit_gpt-inspired hybrid transformer with GatedDeltaNet2)."""

from qwendopamine.models.core.normalization import RMSNorm
from qwendopamine.models.gdn2_gpt.attention import CausalSelfAttention
from qwendopamine.models.gdn2_gpt.block import Block
from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig
from qwendopamine.models.gdn2_gpt.mlp import LLaMAMLP, SwiGLU
from qwendopamine.models.gdn2_gpt.model import GDN2GPT, compute_model_params
from qwendopamine.models.gdn2_gpt.rope import apply_rotary_emb, build_rope_cache

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
