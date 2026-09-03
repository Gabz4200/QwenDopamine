"""GDN-2 GPT Decoder Architecture (lit_gpt-inspired hybrid transformer with GatedDeltaNet2)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from qwendopamine.models.core.normalization import RMSNorm
from qwendopamine.models.gdn2 import GatedDeltaNet2
from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig

RoPECache = tuple[torch.Tensor, torch.Tensor]
KVCache = tuple[torch.Tensor, torch.Tensor]


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: float = 10000.0,
    condense_ratio: float = 1.0,
) -> RoPECache:
    r"""build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: float = 10000.0, condense_ratio: float = 1.0) -> RoPECache

    Build Rotary Position Embedding cos and sin tables.

    Args:
        seq_len (int): Maximum sequence length.
        n_elem (int): Number of embedding elements (must be even).
        dtype (torch.dtype): Output tensor dtype.
        device (torch.device): Output tensor device.
        base (float): RoPE base frequency. Default: ``10000.0``.
        condense_ratio (float): Sequence length compression factor.
            Default: ``1.0``.

    Returns:
        RoPECache: Tuple ``(cos, sin)`` of tensors ``[seq_len, n_elem]``.
    """
    theta = 1.0 / (
        base
        ** (torch.arange(0, n_elem, 2, device=device, dtype=torch.float32) / n_elem)
    )
    seq_idx = torch.arange(seq_len, device=device, dtype=torch.float32) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)
    cos = torch.cos(idx_theta).to(dtype=dtype)
    sin = torch.sin(idx_theta).to(dtype=dtype)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    r"""apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor

    Apply rotary position embedding to input tensor with exact dtype
    preservation.

    Args:
        x (torch.Tensor): Input ``[..., n_elem]``.
        cos (torch.Tensor): Cosine cache ``[seq_len, n_elem]``.
        sin (torch.Tensor): Sine cache ``[seq_len, n_elem]``.

    Returns:
        torch.Tensor: Rotated tensor of the same shape and dtype as ``x``.
    """
    orig_dtype = x.dtype
    x_f = x.float()
    rot_dim = cos.size(-1) * 2
    x_rot = x_f[..., :rot_dim]
    x_pass = x_f[..., rot_dim:]

    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    cos_expanded = cos.float().unsqueeze(0).unsqueeze(2)
    sin_expanded = sin.float().unsqueeze(0).unsqueeze(2)

    y1 = x1 * cos_expanded - x2 * sin_expanded
    y2 = x1 * sin_expanded + x2 * cos_expanded
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    if x_pass.size(-1) > 0:
        y = torch.cat((y, x_pass), dim=-1)
    return y.to(dtype=orig_dtype)


class SwiGLU(nn.Module):
    r"""SwiGLU activation with parallel-gated linear projections.

    Computes ``w3(silu(w1(x)) * w2(x))``.

    Args:
        in_features (int): Input dimension.
        hidden_features (int): Hidden (intermediate) dimension.
        bias (bool): Whether to include bias on linear layers. Default: ``False``.
    """

    def __init__(
        self, in_features: int, hidden_features: int, bias: bool = False
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x: torch.Tensor) -> torch.Tensor

        Args:
            x (torch.Tensor): Input ``[..., in_features]``.

        Returns:
            torch.Tensor: Activated output ``[..., in_features]``.
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = F.silu(x1) * x2
        x = self.w3(x)
        return x


class LLaMAMLP(nn.Module):
    r"""LLaMA-style MLP wrapping a SwiGLU activation.

    Args:
        config (GDN2GPTConfig): Configuration with ``n_embd``,
            ``intermediate_size``, and ``bias``.
    """

    def __init__(self, config: GDN2GPTConfig) -> None:
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd, config.intermediate_size, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x: torch.Tensor) -> torch.Tensor

        Args:
            x (torch.Tensor): Input ``[..., n_embd]``.

        Returns:
            torch.Tensor: Output ``[..., n_embd]``.
        """
        return self.swiglu(x)


class CausalSelfAttention(nn.Module):
    r"""Grouped-query attention with RoPE and KV-cache support.

    Args:
        config (GDN2GPTConfig): Configuration specifying ``n_head``,
            ``n_query_groups``, ``head_size``, ``n_embd``, ``nope``,
            ``rotary_percentage``.
        layer_idx (int): Layer index used for cache disambiguation.
        n_embd (int): Embedding dimension.
        head_size (int | None): Override head dimension. Default: ``None``.
    """

    def __init__(
        self,
        config: GDN2GPTConfig,
        layer_idx: int,
        n_embd: int,
        head_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if head_size is not None:
            self.head_size = head_size
            self.n_head = n_embd // head_size
            self.n_query_groups = self.n_head
        else:
            self.head_size = config.head_size
            self.n_head = config.n_head
            self.n_query_groups = config.n_query_groups

        if self.n_head % self.n_query_groups != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_query_groups ({self.n_query_groups})."
            )

        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache | None = None,
        max_seq_length: int | None = None,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        r"""forward(x, rope=None, max_seq_length=None, mask=None, input_pos=None, kv_cache=None) -> tuple[torch.Tensor, KVCache | None]

        Args:
            x (torch.Tensor): Query ``[B, T, D]``.
            rope (RoPECache | None): ``(cos, sin)`` cache.
            max_seq_length (int | None): Cache window length.
            mask (torch.Tensor | None): Attention mask.
            input_pos (torch.Tensor | None): Positions for cache indexing.
            kv_cache (KVCache | None): ``(k_cache, v_cache)``.

        Returns:
            tuple[torch.Tensor, KVCache | None]: ``(output, kv_cache)``.
        """
        b, t, c = x.size()
        qkv = self.attn(x)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(b, t, self.n_query_groups, total_qkv, self.head_size)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(b, t, self.n_head, self.head_size)
        k = k.reshape(b, t, self.n_query_groups, self.head_size)
        v = v.reshape(b, t, self.n_query_groups, self.head_size)

        if rope is not None and not self.config.nope:
            cos, sin = rope
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            if input_pos is not None:
                if max_seq_length is not None and input_pos[-1] >= max_seq_length:
                    input_pos = torch.tensor(
                        max_seq_length - 1, device=input_pos.device
                    )
                    cache_k = torch.roll(cache_k, -1, dims=1)
                    cache_v = torch.roll(cache_v, -1, dims=1)
                k = cache_k.index_copy_(1, input_pos, k)
                v = cache_v.index_copy_(1, input_pos, v)
                kv_cache = (k, v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_query_groups < self.n_head:
            factor = self.n_head // self.n_query_groups
            k = k.repeat_interleave(factor, dim=1)
            v = v.repeat_interleave(factor, dim=1)

        if mask is not None:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        o = o.transpose(1, 2).contiguous().view(b, t, c)
        o = self.proj(o)
        return o, kv_cache


class Block(nn.Module):
    r"""Transformer block choosing between GDN-2 and standard attention.

    Args:
        config (GDN2GPTConfig): Model configuration.
        layer_idx (int): Index of this layer within the model.
    """

    def __init__(self, config: GDN2GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)

        if config.gdn2_layers is not None:
            self.use_gdn2 = layer_idx in config.gdn2_layers
        elif config.gdn2_per_layer > 0:
            self.use_gdn2 = layer_idx % config.gdn2_per_layer == 0
        else:
            self.use_gdn2 = layer_idx == (config.n_layer // 2)

        if self.use_gdn2:
            self.attn = GatedDeltaNet2(
                hidden_size=config.n_embd,
                num_heads=config.n_head,
                head_dim=config.head_size,
                num_v_heads=config.n_head,
                layer_idx=layer_idx,
                norm_eps=config.norm_eps,
                conv_size=config.conv_size,
                expand_v=config.expand_v,
                use_short_conv=config.use_short_conv,
                chunk_size=config.chunk_size or config.train_chunk_size,
                allow_neg_eigval=config.allow_neg_eigval,
                backend=config.backend,
                fp32_decay=config.fp32_decay,
                compile_backend=config.compile_backend,
            )
        else:
            self.attn = CausalSelfAttention(
                config, layer_idx=layer_idx, n_embd=config.n_embd
            )

        if (
            not config.shared_attention_norm
            and config.mlp
            and not config.parallel_residual
        ):
            self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        if config.mlp:
            self.mlp = LLaMAMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache | None,
        max_seq_length: int,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        r"""forward(x: torch.Tensor, rope: RoPECache | None, max_seq_length: int, mask: torch.Tensor | None = None, input_pos: torch.Tensor | None = None, kv_cache: KVCache | None = None) -> tuple[torch.Tensor, KVCache | None]

        Apply attention (GDN-2 or standard) and MLP sublayer with residuals.

        Args:
            x (torch.Tensor): Input ``[B, T, D]``.
            rope (RoPECache | None): ``(cos, sin)`` cache or ``None``.
            max_seq_length (int): Cache window length.
            mask (torch.Tensor | None): Attention mask.
            input_pos (torch.Tensor | None): Positions for cache indexing.
            kv_cache (KVCache | None): ``(k_cache, v_cache)``.

        Returns:
            tuple[torch.Tensor, KVCache | None]: ``(output, kv_cache)``.
        """
        n_1 = self.norm_1(x)
        if self.use_gdn2:
            h, _, new_kv_cache = self.attn(n_1, attention_mask=None)
        else:
            h, new_kv_cache = self.attn(
                n_1, rope, max_seq_length, mask, input_pos, kv_cache
            )

        if self.config.parallel_residual:
            if self.config.mlp:
                h = h + self.mlp(n_1)
            x = x + h
        else:
            x = x + h
            if self.config.mlp:
                n_2 = self.norm_2(x)
                h = self.mlp(n_2)
                x = x + h
        return x, new_kv_cache


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
        return self.lm_head(x)

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
