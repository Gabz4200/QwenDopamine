"""Kaggle 2xT4 GPU smoke-test + WikiText-2 training for GatedSurpriseNetAdam GPT Model.

Architecture:
  Transformer-based GPT Decoder model (1B scale specification) with
  pre-layer RMSNorm, SwiGLU MLP, RoPE positional embeddings, CausalSelfAttention,
  and GatedSurpriseNetAdam token mixer positioned at the center layer.

Part 1 — Sanity checks & 1B Architecture Inspection
  1. 1B Model Specification & parameter breakdown inspection.
  2. Synthetic overfit on hybrid GPT model with central GatedSurpriseNet block.
  3. Serial-vs-chunk parity: SurpriseMemoryAdam.serial_scan and
     chunk_parallel_training_scan match within tight tolerance.

Part 2 — WikiText-2 LM training
  Load Salesforce/wikitext, build the hybrid Transformer + central
  GatedSurpriseNet GPT model, train for a fixed number of steps with
  DDP + AMP, and log cross-entropy loss, perplexity, and negative
  log-likelihood.

Designed for Kaggle with 2x T4 GPUs. Uses ``torch.distributed`` DDP
when multiple GPUs are detected; falls back to single-GPU/CPU otherwise.

Run on Kaggle::

    python notebooks/test_gated_surprise_net_gpu.py
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import fcntl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from datasets import load_dataset
from matplotlib import ticker
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from transformers import AutoTokenizer

REPO_URL = "https://github.com/Gabz4200/QwenDopamine.git"
PIP_REPO_URL = "git+" + REPO_URL

_LOCK_PATH = os.path.join(tempfile.gettempdir(), "qwendopamine_setup.lock")


def _ensure_dependencies() -> None:
    def _check() -> tuple[bool, bool]:
        need_mask = False
        try:
            import transformers.masking_utils

            need_mask = not hasattr(
                transformers.masking_utils, "create_recurrent_attention_mask"
            )
        except (ImportError, AttributeError):
            need_mask = True

        try:
            importlib.metadata.version("qwendopamine")
            need_qwen = False
        except importlib.metadata.PackageNotFoundError:
            need_qwen = True

        return need_mask, need_qwen

    need_mask, need_qwen = _check()
    if not (need_mask or need_qwen):
        return

    with open(_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            need_mask, need_qwen = _check()
            if not (need_mask or need_qwen):
                return

            to_install: list[str] = []
            if need_mask:
                to_install.append("transformers>=4.49.0")
            if need_qwen:
                to_install.append(PIP_REPO_URL)

            print(
                "[setup] Installing/upgrading dependencies "
                f"({', '.join(to_install)})..."
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade-strategy",
                    "only-if-needed",
                ]
                + to_install,
                check=True,
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


_ensure_dependencies()

from qwendopamine.models.gated_surprise_net import (
    GatedSurpriseNetAdam,
    SurpriseMemoryAdam,
)

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

has_cuda = torch.cuda.is_available()
if has_cuda:
    if WORLD_SIZE > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda", LOCAL_RANK)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    if WORLD_SIZE > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    device = torch.device("cpu")
    dtype = torch.float32

print(
    f"[env] rank={RANK}/{WORLD_SIZE}  local_rank={LOCAL_RANK}  "
    f"device={device}  dtype={dtype}"
)
if has_cuda:
    print(f"[env] GPU: {torch.cuda.get_device_name(LOCAL_RANK)}")

IS_MAIN = RANK == 0

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
    """Build Rotary Position Embedding cos and sin tables."""
    theta = 1.0 / (
        base
        ** (
            torch.arange(0, n_elem, 2, device=device, dtype=torch.float32)
            / n_elem
        )
    )
    seq_idx = (
        torch.arange(seq_len, device=device, dtype=torch.float32)
        / condense_ratio
    )
    idx_theta = torch.outer(seq_idx, theta)
    cos = torch.cos(idx_theta).to(dtype=dtype)
    sin = torch.sin(idx_theta).to(dtype=dtype)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to input tensor with exact dtype preservation."""
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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, bias: bool = False
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = F.silu(x1) * x2
        x = self.w3(x)
        return x


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.swiglu = SwiGLU(
            config.n_embd, config.intermediate_size, bias=config.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: Config,
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
                if (
                    max_seq_length is not None
                    and input_pos[-1] >= max_seq_length
                ):
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
    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)

        if config.surprise_net_layers is not None:
            self.use_surprise_net = layer_idx in config.surprise_net_layers
        elif config.surprise_net_per_layer > 0:
            self.use_surprise_net = (
                layer_idx % config.surprise_net_per_layer == 0
            )
        else:
            # Default: place GatedSurpriseNet at the center layer
            self.use_surprise_net = layer_idx == (config.n_layer // 2)

        if self.use_surprise_net:
            self.attn = GatedSurpriseNetAdam(
                hidden_size=config.n_embd,
                num_heads=config.n_head,
                head_dim=config.head_size,
                num_v_heads=config.n_head,
                layer_idx=layer_idx,
                norm_eps=config.norm_eps,
                conv_size=config.conv_size,
                expand_v=config.expand_v,
                use_short_conv=config.use_short_conv,
                train_chunk_size=config.train_chunk_size,
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
        n_1 = self.norm_1(x)
        if self.use_surprise_net:
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


@dataclass
class Config:
    name: str = "1B"
    block_size: int = 2048
    vocab_size: int = 50257
    padded_vocab_size: int = 50257
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

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Config:
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
        base.update(kwargs)
        return cls(**base)


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=False
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(
                    Block(config, i) for i in range(config.n_layer)
                ),
                ln_f=RMSNorm(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: RoPECache | None = None
        self.mask_cache: torch.Tensor | None = None
        self.kv_caches: list[KVCache | None] = []
        self.max_len = self.config.block_size
        self.mamba_init = config.mamba_init
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
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
                if (
                    module.bias is not None
                    and not getattr(module.bias, "_no_reinit", False)
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
        b, t = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size

        if not self.config.nope:
            if (
                self.rope_cache is None
                or self.rope_cache[0].size(0) < max(t, self.max_len)
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
                assert input_pos is not None
                cos = cos.index_select(0, input_pos)
                sin = sin.index_select(0, input_pos)
            assert self.mask_cache is not None
            assert input_pos is not None
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            if not self.config.nope and cos is not None and sin is not None:
                cos = cos[:t]
                sin = sin[:t]
            mask = None

        rope = None if self.config.nope else (cos, sin)

        x = self.transformer.wte(idx)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, _ = block(x, rope, max_seq_length)
        else:
            start_pos = int(input_pos[0].item()) if input_pos is not None else 0
            if start_pos == 0:
                self.kv_caches = []

            self.kv_caches = self.kv_caches or self.build_kv_caches(
                x, max_seq_length
            )

            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x,
                    rope,
                    max_seq_length,
                    mask,
                    input_pos,
                    self.kv_caches[i],
                )

        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> GPT:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        return build_rope_cache(
            seq_len=seq_len,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.float32,
            device=idx.device,
            base=self.config.rope_base,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(
            (self.config.block_size, self.config.block_size),
            device=idx.device,
            dtype=torch.bool,
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(
        self, idx: torch.Tensor, max_seq_length: int
    ) -> list[KVCache | None]:
        b = idx.size(0)
        heads = self.config.n_query_groups
        k_cache_shape = (b, max_seq_length, heads, self.config.head_size)
        v_cache_shape = (b, max_seq_length, heads, self.config.head_size)
        dev = idx.device

        caches: list[KVCache | None] = []
        for block in self.transformer.h:
            if block.use_surprise_net:
                caches.append(None)
            else:
                caches.append(
                    (
                        torch.zeros(k_cache_shape, device=dev),
                        torch.zeros(v_cache_shape, device=dev),
                    )
                )
        return caches


def compute_model_params(cfg: Config) -> dict[str, int]:
    """Analytically compute model parameter counts across components."""
    embed_params = cfg.padded_vocab_size * cfg.n_embd
    lm_head_params = cfg.n_embd * cfg.padded_vocab_size
    final_norm_params = cfg.n_embd

    mlp_params_per_layer = (cfg.n_embd * cfg.intermediate_size * 2) + (
        cfg.intermediate_size * cfg.n_embd
    )
    norm_params_per_layer = cfg.n_embd * 2

    qkv_dim = (cfg.n_head + 2 * cfg.n_query_groups) * cfg.head_size
    attn_params_per_layer = (cfg.n_embd * qkv_dim) + (cfg.n_embd * cfg.n_embd)

    k_dim = cfg.n_head * cfg.head_size
    v_dim = int(cfg.n_head * cfg.head_size * cfg.expand_v)
    surprise_projs = (cfg.n_embd * k_dim * 2) + (cfg.n_embd * v_dim)
    conv_params = (
        (k_dim * cfg.conv_size * 2) + (v_dim * cfg.conv_size)
        if cfg.use_short_conv
        else 0
    )
    f_proj = (cfg.n_embd * v_dim) + (v_dim * k_dim)
    b_proj = cfg.n_embd * k_dim
    w_proj = cfg.n_embd * v_dim
    g_proj = (cfg.n_embd * v_dim) + (v_dim * v_dim) + v_dim
    o_proj = v_dim * cfg.n_embd
    o_norm = v_dim
    dt_and_a = cfg.n_head + k_dim
    surprise_attn_params = (
        surprise_projs
        + conv_params
        + f_proj
        + b_proj
        + w_proj
        + g_proj
        + o_proj
        + o_norm
        + dt_and_a
    )

    num_surprise_layers = (
        len(cfg.surprise_net_layers)
        if cfg.surprise_net_layers is not None
        else (
            cfg.n_layer // cfg.surprise_net_per_layer
            if cfg.surprise_net_per_layer > 0
            else 1
        )
    )
    num_standard_attn_layers = cfg.n_layer - num_surprise_layers

    total = (
        embed_params
        + lm_head_params
        + final_norm_params
        + (mlp_params_per_layer + norm_params_per_layer) * cfg.n_layer
        + attn_params_per_layer * num_standard_attn_layers
        + surprise_attn_params * num_surprise_layers
    )
    return {
        "total": total,
        "embed": embed_params,
        "lm_head": lm_head_params,
        "standard_block": mlp_params_per_layer
        + norm_params_per_layer
        + attn_params_per_layer,
        "surprise_block": mlp_params_per_layer
        + norm_params_per_layer
        + surprise_attn_params,
        "num_standard_layers": num_standard_attn_layers,
        "num_surprise_layers": num_surprise_layers,
    }


def inspect_1b_architecture() -> None:
    """Inspect and print the 1B scale model specification."""
    cfg_1b = Config.from_name("1B")
    stats = compute_model_params(cfg_1b)
    center_idx = cfg_1b.n_layer // 2

    print("=" * 60)
    print(f"1B Model Specification ({cfg_1b.name})")
    print("=" * 60)
    print(
        f"  Total Parameters:        {stats['total']:,} (~{stats['total']/1e9:.2f}B)"
    )
    print(f"  Layers (n_layer):        {cfg_1b.n_layer}")
    print(f"  Hidden Dim (n_embd):     {cfg_1b.n_embd}")
    print(
        f"  Attention Heads:         {cfg_1b.n_head} (head_size={cfg_1b.head_size})"
    )
    print(f"  GQA KV Groups:           {cfg_1b.n_query_groups}")
    print(f"  SwiGLU Intermediate:     {cfg_1b.intermediate_size}")
    print(f"  Vocabulary Size:         {cfg_1b.vocab_size:,}")
    print(f"  Max Context Length:      {cfg_1b.block_size:,}")
    print(f"  Center Layer Index:      Layer {center_idx} (GatedSurpriseNetAdam)")
    print(
        f"  Standard Layers:         {stats['num_standard_layers']}x CausalSelfAttention"
    )
    print(
        f"  Surprise Net Layers:     {stats['num_surprise_layers']}x GatedSurpriseNetAdam"
    )
    print(f"  Standard Block Params:   {stats['standard_block']:,}")
    print(f"  Center Surprise Block:   {stats['surprise_block']:,}")
    print(f"  Token Embeddings:        {stats['embed']:,}")
    print("=" * 60)


def run_synthetic_overfit() -> tuple[float, float, bool]:
    """Verify learning on a small GPT model with central GatedSurpriseNetAdam."""
    torch.manual_seed(0)
    vocab_size = 64
    seq_len = 32
    batch_size = 4
    synth_steps = 200

    cfg = Config(
        name="synthetic_hybrid",
        block_size=seq_len,
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        head_size=32,
        n_query_groups=4,
        intermediate_size=344,
        norm_eps=1e-5,
        train_chunk_size=seq_len,
    )
    model = GPT(cfg).to(device=device, dtype=dtype)

    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if IS_MAIN:
        print(f"[synth] params: {total_params:,}  trainable: {trainable:,}")
        center_layer = cfg.n_layer // 2
        print(f"[synth] Center GatedSurpriseNet placed at layer {center_layer}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    initial_loss = 0.0
    final_loss_val = 0.0
    for step in range(synth_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.item())
        final_loss_val = float(loss.item())
        if step % 50 == 0 and IS_MAIN:
            print(f"[synth] step {step:03d}  loss={loss.item():.4f}")

    if IS_MAIN:
        print(f"[synth] initial={initial_loss:.4f}  final={final_loss_val:.4f}")

    passed = final_loss_val < initial_loss * 0.5 and math.isfinite(
        final_loss_val
    )
    return initial_loss, final_loss_val, passed


@torch.no_grad()
def run_serial_chunk_parity() -> bool:
    """Verify parity between serial scan and chunk parallel scan."""
    bs, t, h, d = 2, 32, 4, 16
    torch.manual_seed(42)
    q = torch.randn(bs, t, h, d, device=device, dtype=dtype)
    k = torch.randn(bs, t, h, d, device=device, dtype=dtype)
    v = torch.randn(bs, t, h, d, device=device, dtype=dtype)
    g = torch.randn(bs, t, h, d, device=device, dtype=dtype).abs_().mul_(-1)
    b = torch.rand(bs, t, h, d, device=device, dtype=dtype)
    w = torch.rand(bs, t, h, d, device=device, dtype=dtype)

    memory = SurpriseMemoryAdam(num_heads=h, head_k_dim=d, head_v_dim=d).to(
        device=device, dtype=dtype
    )

    out_s, _, nll_s = memory.serial_scan(q, k, v, g, b, w)
    out_c, _, nll_c = memory.chunk_parallel_training_scan(
        q, k, v, g, b, w, chunk_size=16
    )

    passed = torch.allclose(out_s, out_c, atol=1e-3) and torch.allclose(
        nll_s, nll_c, atol=1e-3
    )
    if not passed and IS_MAIN:
        print("[parity] serial vs chunk: FAIL")
        print(f"  output close: {torch.allclose(out_s, out_c, atol=1e-3)}")
        print(f"  nll close: {torch.allclose(nll_s, nll_c, atol=1e-3)}")
    return passed


@dataclass
class WikiTextConfig:
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_seq_len: int = 128
    max_train_examples: int | None = 5000
    max_val_examples: int | None = 1000
    batch_size: int = 16


def load_wikitext_tokenized(
    cfg: WikiTextConfig, tokenizer: Any
) -> tuple[TensorDataset, TensorDataset]:
    if IS_MAIN:
        print(f"[data] Loading {cfg.dataset_name} ({cfg.dataset_config}) ...")
    try:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    except TypeError:
        ds = load_dataset(
            cfg.dataset_name, cfg.dataset_config, trust_remote_code=True
        )

    def encode_split(split: str, max_examples: int | None) -> list[list[int]]:
        texts: list[str] = []
        for ex in ds[split]:
            txt = ex.get("text", "").strip()
            if txt:
                texts.append(txt)
            if max_examples is not None and len(texts) >= max_examples:
                break
        if IS_MAIN:
            print(f"[data] {split}: {len(texts)} passages")
        seqs: list[list[int]] = []
        for txt in texts:
            ids = tokenizer(txt, truncation=False, add_special_tokens=False)[
                "input_ids"
            ]
            if len(ids) < 2:
                continue
            for i in range(0, len(ids) - cfg.max_seq_len, cfg.max_seq_len):
                window = ids[i : i + cfg.max_seq_len + 1]
                if len(window) == cfg.max_seq_len + 1:
                    seqs.append(window)
        return seqs

    train_seqs = encode_split("train", cfg.max_train_examples)
    val_seqs = encode_split("validation", cfg.max_val_examples)

    if not train_seqs:
        raise RuntimeError("No training sequences produced.")

    def to_tensor(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.tensor([s[:-1] for s in seqs], dtype=torch.long)
        targets = torch.tensor([s[1:] for s in seqs], dtype=torch.long)
        return inputs, targets

    train_in, train_tgt = to_tensor(train_seqs)
    val_in, val_tgt = to_tensor(val_seqs)
    if IS_MAIN:
        print(
            f"[data] train sequences: {train_in.shape[0]}  "
            f"val sequences: {val_in.shape[0]}"
        )
    return TensorDataset(train_in, train_tgt), TensorDataset(val_in, val_tgt)


@dataclass
class TrainConfig:
    num_steps: int = 1000
    log_interval: int = 50
    eval_interval: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    use_amp: bool = True
    chunk_size: int = 128


def evaluate(
    model: nn.Module,
    val_dl: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    nll_total = 0.0
    eval_t0 = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(val_dl):
            if max_batches is not None and batch_idx >= max_batches:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda"),
            ):
                logits = model(xb)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            batch_tokens = yb.numel()
            total_loss += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens
            nll_total += float(loss.item()) * batch_tokens

    eval_time = time.perf_counter() - eval_t0

    if WORLD_SIZE > 1:
        stats = torch.tensor(
            [total_loss, total_tokens, nll_total], device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_tokens, nll_total = stats.tolist()

    avg_loss = total_loss / max(total_tokens, 1.0)
    perplexity = math.exp(min(avg_loss, 50))
    tps = total_tokens / max(eval_time, 1e-9)
    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
        "val_nll": nll_total,
        "val_tokens": total_tokens,
        "val_tps": tps,
    }


def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, list[float]]:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    try:
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=(
                device.type == "cuda"
                and dtype == torch.float16
                and cfg.use_amp
            ),
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=(
                device.type == "cuda"
                and dtype == torch.float16
                and cfg.use_amp
            )
        )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_steps": [],
        "val_loss": [],
        "val_perplexity": [],
        "val_nll": [],
        "val_tps": [],
        "val_steps": [],
        "tokens_seen": [],
        "lr": [],
        "step_time_s": [],
    }
    tokens_seen = 0
    t0 = time.perf_counter()
    step = 0
    epoch = 0

    while step < cfg.num_steps:
        if isinstance(train_dl.sampler, DistributedSampler):
            train_dl.sampler.set_epoch(epoch)
        epoch += 1

        for xb, yb in train_dl:
            if step >= cfg.num_steps:
                break

            xb = xb.to(device)
            yb = yb.to(device)

            if step < cfg.warmup_steps:
                lr = cfg.lr * (step + 1) / cfg.warmup_steps
            else:
                progress = (step - cfg.warmup_steps) / max(
                    cfg.num_steps - cfg.warmup_steps, 1
                )
                lr = cfg.lr * (0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda"),
            ):
                logits = model(xb)
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tokens_seen += yb.numel()
            step_time = time.perf_counter() - t0

            if step % cfg.log_interval == 0 and IS_MAIN:
                history["train_loss"].append(float(loss.item()))
                history["train_steps"].append(float(step))
                history["tokens_seen"].append(float(tokens_seen))
                history["lr"].append(lr)
                history["step_time_s"].append(step_time)
                print(
                    f"[train] step {step:5d}  loss={loss.item():.4f}  "
                    f"lr={lr:.2e}  tok/s={tokens_seen / max(step_time, 1e-9):.0f}"
                )

            if step % cfg.eval_interval == 0 and step > 0:
                metrics = evaluate(
                    model, val_dl, loss_fn, device, dtype, max_batches=50
                )
                if IS_MAIN:
                    history["val_loss"].append(metrics["val_loss"])
                    history["val_perplexity"].append(metrics["val_perplexity"])
                    history["val_nll"].append(metrics["val_nll"])
                    history["val_tps"].append(metrics["val_tps"])
                    history["val_steps"].append(float(step))
                    print(
                        f"[eval]  step {step:5d}  loss={metrics['val_loss']:.4f}  "
                        f"ppl={metrics['val_perplexity']:.2f}  "
                        f"nll={metrics['val_nll']:.2f}  "
                        f"tps={metrics['val_tps']:.0f}"
                    )
                model.train()

            step += 1

    metrics = evaluate(model, val_dl, loss_fn, device, dtype)
    if IS_MAIN:
        history["val_loss"].append(metrics["val_loss"])
        history["val_perplexity"].append(metrics["val_perplexity"])
        history["val_nll"].append(metrics["val_nll"])
        history["val_tps"].append(metrics["val_tps"])
        history["val_steps"].append(float(step))
        print(
            f"[final] loss={metrics['val_loss']:.4f}  "
            f"ppl={metrics['val_perplexity']:.2f}  "
            f"nll={metrics['val_nll']:.2f}  "
            f"tps={metrics['val_tps']:.0f}"
        )
    return history


def plot_metrics(
    history: dict[str, list[float]], save_path: str = "metrics.png"
) -> None:
    if not IS_MAIN:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "GatedSurpriseNetAdam Hybrid GPT — WikiText-2 Training Metrics",
        fontsize=13,
    )

    plots = [
        (
            axes[0, 0],
            "train_steps",
            "train_loss",
            "Train Loss",
            "Loss",
            "tab:blue",
        ),
        (axes[0, 1], "val_steps", "val_loss", "Val Loss", "Loss", "tab:orange"),
        (
            axes[0, 2],
            "val_steps",
            "val_perplexity",
            "Val Perplexity",
            "Perplexity",
            "tab:green",
        ),
        (axes[1, 0], "val_steps", "val_nll", "Val NLL", "NLL", "tab:red"),
        (
            axes[1, 1],
            "val_steps",
            "val_tps",
            "Val Throughput",
            "Tokens / sec",
            "tab:purple",
        ),
        (axes[1, 2], "train_steps", "lr", "Learning Rate", "LR", "tab:brown"),
    ]

    for ax, x_key, y_key, title, ylabel, color in plots:
        x_data = history.get(x_key, [])
        y_data = history.get(y_key, [])
        if x_data and y_data and len(x_data) == len(y_data):
            ax.plot(x_data, y_data, marker="o", color=color)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.grid(True)

    axes[1, 2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[plot] Saved metrics plot to {save_path}")
    plt.close(fig)


def main() -> None:
    if IS_MAIN:
        inspect_1b_architecture()

    sanity_ok = True

    if IS_MAIN:
        print("\n===== Sanity check 1: synthetic overfit =====")
    initial_loss, final_loss, synth_ok = run_synthetic_overfit()
    if not synth_ok:
        sanity_ok = False
        if IS_MAIN:
            print(
                f"[check] Synthetic overfit: FAIL — loss {initial_loss:.4f} -> {final_loss:.4f}"
            )
    elif IS_MAIN:
        print("[check] Synthetic overfit: PASS")

    if IS_MAIN:
        print("\n===== Sanity check 2: serial vs chunk parity =====")
    parity_ok = run_serial_chunk_parity()
    if not parity_ok:
        sanity_ok = False
        if IS_MAIN:
            print("[check] Serial vs chunk parity: FAIL")
    elif IS_MAIN:
        print("[check] Serial vs chunk parity: PASS")

    if WORLD_SIZE > 1:
        sanity_tensor = torch.tensor([1 if sanity_ok else 0], device=device)
        dist.broadcast(sanity_tensor, src=0)
        sanity_ok = bool(sanity_tensor.item())

    if IS_MAIN:
        if sanity_ok:
            print(
                "[check] All sanity checks passed. Proceeding to WikiText-2 training."
            )
        else:
            print(
                "[check] One or more sanity checks failed. Skipping WikiText-2 training."
            )

    if not sanity_ok:
        if WORLD_SIZE > 1:
            dist.destroy_process_group()
        return

    if IS_MAIN:
        print("\n===== Part 2: WikiText-2 training =====")

    wt_cfg = WikiTextConfig(
        max_seq_len=128,
        max_train_examples=5000,
        max_val_examples=1000,
        batch_size=16,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_wikitext_tokenized(wt_cfg, tokenizer)

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True) if WORLD_SIZE > 1 else None
    )
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False) if WORLD_SIZE > 1 else None
    )

    pin_memory = device.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=wt_cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=wt_cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
    )

    train_cfg = TrainConfig(
        num_steps=1000,
        log_interval=50,
        eval_interval=200,
        lr=3e-4,
        weight_decay=0.1,
        warmup_steps=100,
        chunk_size=128,
    )

    lm_cfg = Config(
        name="hybrid_lm",
        block_size=max(wt_cfg.max_seq_len, 2048),
        vocab_size=tokenizer.vocab_size,
        padded_vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=256,
        head_size=64,
        n_query_groups=4,
        intermediate_size=688,
        norm_eps=1e-5,
        train_chunk_size=train_cfg.chunk_size,
    )

    model = GPT(lm_cfg).to(device=device, dtype=dtype)
    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    total_params = sum(p.numel() for p in model.parameters())
    if IS_MAIN:
        center_layer = lm_cfg.n_layer // 2
        print(
            f"[model] GPT Hybrid LM  params={total_params:,}  "
            f"layers={lm_cfg.n_layer} (Center SurpriseNet at layer {center_layer})"
        )

    history = train(model, train_dl, val_dl, train_cfg, device, dtype)

    plot_metrics(history, save_path="metrics.png")

    if WORLD_SIZE > 1:
        dist.barrier()

    if IS_MAIN:
        print("\n===== Summary =====")
        final_train = (
            history["train_loss"][-1]
            if history["train_loss"]
            else float("nan")
        )
        final_ppl = (
            history["val_perplexity"][-1]
            if history["val_perplexity"]
            else float("nan")
        )
        final_nll = (
            history["val_nll"][-1] if history["val_nll"] else float("nan")
        )
        final_tps = (
            history["val_tps"][-1] if history["val_tps"] else float("nan")
        )
        print(f"  Train loss (last logged): {final_train:.4f}")
        print(f"  Val perplexity:          {final_ppl:.2f}")
        print(f"  Val NLL:                 {final_nll:.2f}")
        print(f"  Val throughput:          {final_tps:.0f} tok/s")
        print(f"  Params:                  {total_params:,}")
        print(f"  GPUs:                    {WORLD_SIZE}x T4")
        print("  Data:                    Salesforce/wikitext-2-raw-v1")
        print(
            f"  Architecture:            GPT Hybrid (Layers: {lm_cfg.n_layer}, Center: GatedSurpriseNetAdam)"
        )
        print(f"  Train steps:             {train_cfg.num_steps}")
        print("  Result: Hybrid GPT LM training complete.")

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
