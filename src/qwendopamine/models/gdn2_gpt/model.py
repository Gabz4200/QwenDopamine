"""GDN-2 GPT Decoder Architecture (lit_gpt-inspired hybrid transformer with GatedDeltaNet2).

The helper modules live in dedicated siblings:

  - :mod:`.rope`      — RoPE cache + :func:`apply_rotary_emb`
  - :mod:`.mlp`       — :class:`SwiGLU`, :class:`LLaMAMLP`
  - :mod:`.attention` — :class:`CausalSelfAttention`
  - :mod:`.block`     — :class:`Block`

This module owns the top-level :class:`GDN2GPT` model and the
:func:`compute_model_params` helper.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from qwendopamine.models.core.normalization import RMSNorm
from qwendopamine.models.gdn2_gpt.block import Block
from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig
from qwendopamine.models.gdn2_gpt.rope import (
    KVCache,
    RoPECache,
    build_rope_cache,
)


class GDN2GPT(nn.Module):
    r"""GDN2GPT(config: GDN2GPTConfig) -> None

    Lit-GPT style decoder with hybrid GDN-2 / standard-attention blocks.

    Args:
        config (GDN2GPTConfig): Model configuration.
    """

    def __init__(self, config: GDN2GPTConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

        self.rope_cache: RoPECache | None = None
        self.mask_cache: torch.Tensor | None = None
        self.kv_caches: list[KVCache | None] = []
        self.max_len = self.config.block_size
        self.mamba_init = config.mamba_init
        self.apply(self._init_weights)

    @property
    def transformer(self) -> nn.ModuleDict:
        r"""transformer() -> nn.ModuleDict

        Return the transformer submodules (embeddings, blocks, norm).

        Returns:
            nn.ModuleDict: Named submodules ``wte``, ``h``, ``ln_f``.
        """
        return nn.ModuleDict({"wte": self.wte, "h": self.h, "ln_f": self.ln_f})

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 / Mamba init conventions."""
        if isinstance(module, nn.Embedding):
            if self.mamba_init:
                torch.nn.init.normal_(module.weight, std=0.02)
            else:
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=math.sqrt(2.0 / 5 / self.config.n_embd),
                )
        elif isinstance(module, nn.Linear):
            if self.mamba_init:
                if module.bias is not None and not getattr(
                    module.bias, "_no_reinit", False
                ):
                    nn.init.zeros_(module.bias)
            else:
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=math.sqrt(2.0 / 5 / self.config.n_embd),
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def reset_cache(self) -> None:
        """reset_cache() -> None

        Clear KV caches and reset Rope/Mask caches to ``None``.

        Returns:
            None
        """
        self.max_len = self.config.block_size
        self.kv_caches.clear()
        self.rope_cache = None
        self.mask_cache = None

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: int | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""forward(idx: torch.Tensor, max_seq_length: int | None = None, input_pos: torch.Tensor | None = None) -> torch.Tensor

        Forward pass through embedding, transformer blocks, and LM head.

        Args:
            idx (torch.Tensor): Token ids ``[B, T]``.
            max_seq_length (int | None): Maximum sequence length for cache.
                Default: ``None`` (uses ``config.block_size``).
            input_pos (torch.Tensor | None): Input positions ``[T]`` for
                incremental decoding. Default: ``None``.

        Returns:
            torch.Tensor: Vocabulary logits ``[B, T, vocab_size]``.
        """
        _b, t = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size

        if not self.config.nope:
            if self.rope_cache is None or self.rope_cache[0].size(0) < max(
                t, self.max_len
            ):
                self.max_len = max(t, self.max_len)
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            cos, sin = self.rope_cache
        else:
            cos, sin = None, None

        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if use_kv_cache:
            if not self.config.nope and cos is not None and sin is not None:
                if input_pos is None:
                    raise ValueError(
                        "input_pos is required when using KV cache with RoPE."
                    )
                cos = cos.index_select(0, input_pos)
                sin = sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise RuntimeError("mask_cache is required when using KV cache.")
            if input_pos is None:
                raise ValueError("input_pos is required when using KV cache.")
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            if not self.config.nope and cos is not None and sin is not None:
                cos = cos[:t]
                sin = sin[:t]
            mask = None

        rope = None if self.config.nope else (cos, sin)

        x = self.wte(idx)

        if not use_kv_cache:
            if self.config.gradient_checkpointing and self.training:
                for block in self.h:
                    x, _ = torch.utils.checkpoint.checkpoint(
                        block,
                        x,
                        rope,
                        max_seq_length,
                        use_reentrant=False,
                    )
            else:
                for block in self.h:
                    x, _ = block(x, rope, max_seq_length)
        else:
            start_pos = int(input_pos[0].item()) if input_pos is not None else 0
            if start_pos == 0:
                self.kv_caches = []

            self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length)

            for i, block in enumerate(self.h):
                x, self.kv_caches[i] = block(
                    x,
                    rope,
                    max_seq_length,
                    mask,
                    input_pos,
                    self.kv_caches[i],
                )

        x = self.ln_f(x)
        result: torch.Tensor = self.lm_head(x)
        return result

    def gradient_checkpointing_enable(self) -> None:
        """gradient_checkpointing_enable() -> None

        Enable gradient checkpointing on forward passes.

        Returns:
            None
        """
        self.config.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """gradient_checkpointing_disable() -> None

        Disable gradient checkpointing.

        Returns:
            None
        """
        self.config.gradient_checkpointing = False

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> GDN2GPT:
        r"""from_name(name: str, **kwargs: Any) -> GDN2GPT

        Instantiate from a preset config name.

        Args:
            name (str): Preset name (e.g. ``"1B"``).
            **kwargs: Extra keyword arguments forwarded to
                :meth:`GDN2GPTConfig.from_name`.

        Returns:
            GDN2GPT: Instantiated model.
        """
        return cls(GDN2GPTConfig.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        r"""build_rope_cache(idx: torch.Tensor, seq_len: int) -> RoPECache

        Build (or rebuild) the RoPE cos/sin cache.

        Args:
            idx (torch.Tensor): Reference tensor providing device and dtype.
            seq_len (int): Maximum sequence length.

        Returns:
            RoPECache: ``(cos, sin)`` tensors ``[seq_len, n_elem]``.
        """
        return build_rope_cache(
            seq_len=seq_len,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.float32,
            device=idx.device,
            base=self.config.rope_base,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        r"""build_mask_cache(idx: torch.Tensor) -> torch.Tensor

        Build a lower-triangular causal mask cache.

        Args:
            idx (torch.Tensor): Reference tensor providing the device.

        Returns:
            torch.Tensor: Boolean mask ``[1, 1, B, B]`` (tril).
        """
        ones = torch.ones(
            (self.config.block_size, self.config.block_size),
            device=idx.device,
            dtype=torch.bool,
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(
        self, idx: torch.Tensor, max_seq_length: int
    ) -> list[KVCache | None]:
        r"""build_kv_caches(idx: torch.Tensor, max_seq_length: int) -> list[KVCache | None]

        Pre-allocate per-layer KV caches for decoding.

        Args:
            idx (torch.Tensor): Reference tensor providing batch size and
                device.
            max_seq_length (int): Maximum cache length per layer.

        Returns:
            list[KVCache | None]: One ``(k_cache, v_cache)`` tuple per block,
            or ``None`` for GDN-2 blocks that manage their own state.
        """
        b = idx.size(0)
        heads = self.config.n_query_groups
        k_cache_shape = (b, max_seq_length, heads, self.config.head_size)
        v_cache_shape = (b, max_seq_length, heads, self.config.head_size)
        dev = idx.device

        caches: list[KVCache | None] = []
        for block in self.h:
            if block.use_gdn2:
                caches.append(None)
            else:
                caches.append(
                    (
                        torch.zeros(k_cache_shape, device=dev),
                        torch.zeros(v_cache_shape, device=dev),
                    )
                )
        return caches


def compute_model_params(cfg: GDN2GPTConfig) -> dict[str, int]:
    r"""compute_model_params(cfg: GDN2GPTConfig) -> dict[str, int]

    Analytically compute model parameter counts across components.

    Args:
        cfg (GDN2GPTConfig): Model configuration.

    Returns:
        dict[str, int]: Per-component param counts keyed by ``"total"``,
        ``"embed"``, ``"lm_head"``, ``"standard_block"``, ``"gdn2_block"``,
        ``"num_standard_layers"``, ``"num_gdn2_layers"``.
    """
    padded_vocab = cfg.padded_vocab_size or cfg.vocab_size
    embed_params = padded_vocab * cfg.n_embd
    lm_head_params = cfg.n_embd * padded_vocab
    final_norm_params = cfg.n_embd

    mlp_params_per_layer = (cfg.n_embd * cfg.intermediate_size * 2) + (
        cfg.intermediate_size * cfg.n_embd
    )
    norm_params_per_layer = cfg.n_embd * 2

    qkv_dim = (cfg.n_head + 2 * cfg.n_query_groups) * cfg.head_size
    attn_params_per_layer = (cfg.n_embd * qkv_dim) + (cfg.n_embd * cfg.n_embd)

    k_dim = cfg.n_head * cfg.head_size
    hv = int(cfg.head_size * cfg.expand_v)
    v_dim = int(cfg.n_head * hv)
    gdn2_projs = (cfg.n_embd * k_dim * 2) + (cfg.n_embd * v_dim)
    conv_params = (
        (k_dim * cfg.conv_size * 2) + (v_dim * cfg.conv_size)
        if cfg.use_short_conv
        else 0
    )
    f_proj = (cfg.n_embd * hv) + (hv * k_dim)
    b_proj = cfg.n_embd * k_dim
    w_proj = cfg.n_embd * v_dim
    g_proj = (cfg.n_embd * hv) + (hv * v_dim) + v_dim
    o_proj = v_dim * cfg.n_embd
    o_norm = hv
    dt_and_a = cfg.n_head + k_dim
    gdn2_attn_params = (
        gdn2_projs
        + conv_params
        + f_proj
        + b_proj
        + w_proj
        + g_proj
        + o_proj
        + o_norm
        + dt_and_a
    )

    num_gdn2_layers = (
        len(cfg.gdn2_layers)
        if cfg.gdn2_layers is not None
        else (cfg.n_layer // cfg.gdn2_per_layer if cfg.gdn2_per_layer > 0 else 1)
    )
    num_standard_attn_layers = cfg.n_layer - num_gdn2_layers

    total = (
        embed_params
        + lm_head_params
        + final_norm_params
        + (mlp_params_per_layer + norm_params_per_layer) * cfg.n_layer
        + attn_params_per_layer * num_standard_attn_layers
        + gdn2_attn_params * num_gdn2_layers
    )
    return {
        "total": total,
        "embed": embed_params,
        "lm_head": lm_head_params,
        "standard_block": mlp_params_per_layer
        + norm_params_per_layer
        + attn_params_per_layer,
        "gdn2_block": mlp_params_per_layer + norm_params_per_layer + gdn2_attn_params,
        "num_standard_layers": num_standard_attn_layers,
        "num_gdn2_layers": num_gdn2_layers,
    }
