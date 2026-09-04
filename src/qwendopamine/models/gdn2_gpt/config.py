"""Configuration for GDN-2 GPT Decoder Architecture (lit_gpt-inspired hybrid transformer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GDN2GPTConfig:
    r"""GDN2GPTConfig: configuration for the lit_gpt-style hybrid transformer decoder.

    Args:
        name (str): Preset variant name. Default: ``"1B"``.
        block_size (int): Sequence length context. Default: ``2048``.
        vocab_size (int): Vocabulary size. Default: ``50257``.
        padded_vocab_size (int | None): Padded vocab for tensor-parallel.
            Default: ``None`` (defaults to ``vocab_size``).
        n_layer (int): Number of transformer layers. Default: ``24``.
        n_head (int): Number of attention heads. Default: ``16``.
        n_embd (int): Embedding width. Default: ``2048``.
        head_size (int): Per-head dimension. Default: ``128``.
        n_query_groups (int): KV head groups (GQA). Default: ``8``.
        intermediate_size (int): MLP hidden size. Default: ``5504``.
        norm_eps (float): LayerNorm epsilon. Default: ``1e-5``.
        bias (bool): Use bias in linear layers. Default: ``False``.
        nope (bool): Disable positional embeddings. Default: ``False``.
        rotary_percentage (float): RoPE fraction. Default: ``1.0``.
        rope_base (float): RoPE base frequency. Default: ``10000.0``.
        condense_ratio (float): NTK scale. Default: ``1.0``.
        gdn2_layers (list[int] | None): Layer indices using GDN-2.
            Default: ``None``.
        gdn2_per_layer (int): Per-layer GDN flag. Default: ``0``.
        mlp (bool): Include MLP in GDN-2 blocks. Default: ``True``.
        parallel_residual (bool): Use parallel residual. Default: ``False``.
        shared_attention_norm (bool): Share attention norm. Default: ``False``.
        mamba_init (bool): Use Mamba-style init. Default: ``False``.
        conv_size (int): Short convolution kernel. Default: ``4``.
        expand_v (float): Value expansion ratio. Default: ``1.0``.
        use_short_conv (bool): Use short convolution. Default: ``True``.
        train_chunk_size (int): Training chunk. Default: ``128``.
        gradient_checkpointing (bool): Enable gradient checkpointing.
            Default: ``False``.
        chunk_size (int | None): Inference chunk. Default: ``None``
            (defaults to ``train_chunk_size``).
        allow_neg_eigval (bool): Allow negative eigenvalues. Default: ``False``.
        backend (str): Kernel backend. Default: ``"auto"``.
        fp32_decay (bool): FP32 decay in optimizer. Default: ``True``.
        compile_backend (bool): Compile Taichi backend. Default: ``False``.
    """

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
    gdn2_layers: list[int] | None = None
    gdn2_per_layer: int = 0
    mlp: bool = True
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    mamba_init: bool = False
    conv_size: int = 4
    expand_v: float = 1.0
    use_short_conv: bool = True
    train_chunk_size: int = 128
    gradient_checkpointing: bool = False
    chunk_size: int | None = None
    allow_neg_eigval: bool = False
    backend: str = "auto"
    fp32_decay: bool = True
    compile_backend: bool = False

    def __post_init__(self) -> None:
        if self.padded_vocab_size is None:
            self.padded_vocab_size = self.vocab_size
        if self.chunk_size is None:
            self.chunk_size = self.train_chunk_size

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> GDN2GPTConfig:
        r"""from_name(cls, name: str, **kwargs: Any) -> GDN2GPTConfig

        Build a config from a named preset.

        Args:
            name (str): Preset name (e.g. ``"1B"``, ``"3B"``).
            **kwargs (Any): Override preset values.

        Returns:
            GDN2GPTConfig: Configured instance.
        """
        # Common fields shared by every preset. Per-preset entries
        # below only need to specify what differs from the defaults.
        common: dict[str, Any] = {
            "vocab_size": 50257,
            "padded_vocab_size": 50257,
            "norm_eps": 1e-5,
        }
        presets: dict[str, dict[str, Any]] = {
            "1B": {
                **common,
                "name": "1B",
                "block_size": 2048,
                "n_layer": 24,
                "n_head": 16,
                "n_embd": 2048,
                "head_size": 128,
                "n_query_groups": 8,
                "intermediate_size": 5504,
            },
            "1B_mha": {
                **common,
                "name": "1B_mha",
                "block_size": 2048,
                "n_layer": 24,
                "n_head": 16,
                "n_embd": 2048,
                "head_size": 128,
                "n_query_groups": 16,
                "intermediate_size": 5504,
            },
            "small": {
                **common,
                "name": "small",
                "block_size": 512,
                "n_layer": 6,
                "n_head": 8,
                "n_embd": 512,
                "head_size": 64,
                "n_query_groups": 8,
                "intermediate_size": 1376,
            },
            "tiny": {
                **common,
                "name": "tiny",
                "block_size": 256,
                "n_layer": 4,
                "n_head": 4,
                "n_embd": 256,
                "head_size": 64,
                "n_query_groups": 4,
                "intermediate_size": 688,
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
