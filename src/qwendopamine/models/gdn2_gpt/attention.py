"""Causal self-attention with RoPE and KV-cache support.

Moved from ``model.py`` for size.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig
from qwendopamine.models.gdn2_gpt.rope import KVCache, RoPECache, apply_rotary_emb


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
        r"""__init__(config: GDN2GPTConfig, layer_idx: int, n_embd: int, head_size: int | None = None) -> None

        Initialize the attention layer with Q/K/V/output projections and an
        optional QK-norm.

        Args:
            config (GDN2GPTConfig): Model configuration.
            layer_idx (int): Layer index.
            n_embd (int): Embedding dimension.
            head_size (int | None): Override head dimension. Default: ``None``.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = n_embd
        self.n_query_groups = max(config.n_query_groups, 1)
        self.head_size = head_size if head_size is not None else config.head_size
        self.rotary_percentage = config.rotary_percentage

        kv_size = self.n_query_groups * self.head_size
        q_size = self.n_head * self.head_size
        self.q_proj = nn.Linear(self.n_embd, q_size, bias=False)
        self.k_proj = nn.Linear(self.n_embd, kv_size, bias=False)
        self.v_proj = nn.Linear(self.n_embd, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, self.n_embd, bias=False)
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_size, eps=config.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_size, eps=config.norm_eps)

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

        Apply grouped-query attention with optional KV-cache.

        Args:
            x (torch.Tensor): Input ``[B, T, n_embd]``.
            rope (RoPECache | None): RoPE cache.
            max_seq_length (int): Maximum sequence length.
            mask (torch.Tensor | None): Attention mask.
            input_pos (torch.Tensor | None): Cache positions.
            kv_cache (KVCache | None): Optional cache.

        Returns:
            tuple[torch.Tensor, KVCache | None]: Output and updated cache.
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = (
            self.k_proj(x)
            .view(B, T, self.n_query_groups, self.head_size)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, T, self.n_query_groups, self.head_size)
            .transpose(1, 2)
        )

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope is not None and self.rotary_percentage < 1.0:
            rot_dim = int(self.head_size * self.rotary_percentage)
            q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
            k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            q_rot = apply_rotary_emb(q_rot, *rope)
            k_rot = apply_rotary_emb(k_rot, *rope)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        elif rope is not None:
            q = apply_rotary_emb(q, *rope)
            k = apply_rotary_emb(k, *rope)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k = cache_k.to(k.dtype)
            cache_v = cache_v.to(v.dtype)
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        if input_pos is not None:
            k = k[:, :, input_pos]
            v = v[:, :, input_pos]

        if self.n_query_groups != self.n_head:
            repeat_factor = self.n_head // self.n_query_groups
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        scale = 1.0 / math.sqrt(self.head_size)
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        o = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        o = o.transpose(1, 2).contiguous().view(B, T, -1)
        o = self.o_proj(o)
        return o, new_cache
