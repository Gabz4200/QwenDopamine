"""Configuration for Hybrid GPT model with center GatedSurpriseNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SurpriseGPTConfig:
    name: str = "1B"
    block_size: int = 2048
    vocab_size: int = 50257
    padded_vocab_size: int | None = None
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 2048
    head_size: int = 128
    n_query_groups: int = 8
    intermediate_size: int = 5504
    norm_eps: float = 1e-5
    bias: bool = False
    nope: bool = False
    rotary_percentage: float = 1.0
    rope_base: float = 10000.0
    condense_ratio: float = 1.0
    surprise_net_layers: list[int] | None = None
    surprise_net_per_layer: int = 0
    mlp: bool = True
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    mamba_init: bool = False
    conv_size: int = 4
    expand_v: float = 1.0
    use_short_conv: bool = True
    train_chunk_size: int = 128
    mixer_type: str = "surprise"
    gradient_checkpointing: bool = False

    # GDN-2 execution options (passed through to `GatedDeltaNet2`).
    chunk_size: int | None = None
    allow_neg_eigval: bool = False
    backend: str = "auto"
    fp32_decay: bool = False
    compile_backend: bool = False

    def __post_init__(self) -> None:
        if self.padded_vocab_size is None:
            self.padded_vocab_size = self.vocab_size
        if self.chunk_size is None:
            self.chunk_size = self.train_chunk_size

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> SurpriseGPTConfig:
        presets: dict[str, dict[str, Any]] = {
            "1B": {
                "name": "1B",
                "block_size": 2048,
                "vocab_size": 50257,
                "padded_vocab_size": 50257,
                "n_layer": 24,
                "n_head": 16,
                "n_embd": 2048,
                "head_size": 128,
                "n_query_groups": 8,
                "intermediate_size": 5504,
                "norm_eps": 1e-5,
            },
            "1B_mha": {
                "name": "1B_mha",
                "block_size": 2048,
                "vocab_size": 50257,
                "padded_vocab_size": 50257,
                "n_layer": 24,
                "n_head": 16,
                "n_embd": 2048,
                "head_size": 128,
                "n_query_groups": 16,
                "intermediate_size": 5504,
                "norm_eps": 1e-5,
            },
            "small": {
                "name": "small",
                "block_size": 512,
                "vocab_size": 50257,
                "padded_vocab_size": 50257,
                "n_layer": 6,
                "n_head": 8,
                "n_embd": 512,
                "head_size": 64,
                "n_query_groups": 8,
                "intermediate_size": 1376,
                "norm_eps": 1e-5,
            },
            "tiny": {
                "name": "tiny",
                "block_size": 256,
                "vocab_size": 50257,
                "padded_vocab_size": 50257,
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 256,
                "head_size": 64,
                "n_query_groups": 4,
                "intermediate_size": 688,
                "norm_eps": 1e-5,
            },
        }
        if name not in presets:
            raise KeyError(
                f"Unknown config name '{name}'. Available: {list(presets.keys())}"
            )
        base = presets[name].copy()
        if "padded_vocab_size" not in kwargs and "vocab_size" in kwargs:
            base["padded_vocab_size"] = kwargs["vocab_size"]
        base.update(kwargs)
        return cls(**base)
