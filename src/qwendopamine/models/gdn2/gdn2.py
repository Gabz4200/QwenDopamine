# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""GDN-2 (Gated DeltaNet 2) hardware-agnostic token-mixing layer.

This module defines `GatedDeltaNet2`, the `nn.Module` that wraps the GDN-2
recurrence into a drop-in token mixer for Qwen-style Transformer blocks. It
supports both GPU (accelerated via Triton/FLA when available) and CPU/device-agnostic
execution via pure PyTorch reference fallbacks.

GDN-2 extends KDA's scalar-beta erase gate to channel-wise erase (`b`) and write (`w`)
gates:

    S_t = (I - k_t (b_t \odot k_t)^T) \text{Diag}(\exp(g_t)) S_{t-1} + k_t (w_t \odot v_t)^T
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from transformers.cache_utils import Cache

try:
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
except ImportError:
    LinearAttentionCacheLayerMixin = type(None)  # type: ignore[misc, assignment]

# Module-level single-warning guard for CPU fallback
_WARNED_FALLBACKS: set[str] = set()


def _warn_fallback_once(reason: str) -> None:
    if reason not in _WARNED_FALLBACKS:
        _WARNED_FALLBACKS.add(reason)
        warnings.warn(f"[gdn2] Using pure PyTorch fallback: {reason}", stacklevel=2)


# Safe optional Triton/FLA ops imports
_HAS_TRITON_OPS = False
try:
    from .gdn2_ops.chunk_gdn2 import chunk_gdn2 as _triton_chunk_gdn2
    from .gdn2_ops.fused_recurrent_gdn2 import (
        fused_recurrent_gdn2 as _triton_fused_recurrent_gdn2,
    )

    _HAS_TRITON_OPS = True
except ImportError as e:
    _triton_chunk_gdn2 = None
    _triton_fused_recurrent_gdn2 = None
    _warn_fallback_once(f"Triton ops failed to load: {e}")


# Pure PyTorch reference functions for GDN-2 recurrence


def torch_recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Pure PyTorch token-by-token GDN-2 recurrence loop.

    Args:
        q: Query tensor of shape `[B, T, H, d_k]`.
        k: Key tensor of shape `[B, T, H, d_k]`.
        v: Value tensor of shape `[B, T, H, d_v]`.
        g: Log-decay tensor of shape `[B, T, H, d_k]`.
        b: Erase gate tensor of shape `[B, T, H, d_k]`.
        w: Write gate tensor of shape `[B, T, H, d_v]`.
        initial_state: Optional recurrent state tensor of shape `[B, H, d_k, d_v]`.
        output_final_state: Whether to return the final state tensor.
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to queries and keys.

    Returns:
        A tuple `(out, final_state)` where `out` has shape `[B, T, H, d_v]`.
    """
    batch_size, seq_len, num_heads, d_k = q.shape
    d_v = v.shape[-1]

    dtype = q.dtype
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    b_f = b.float()
    w_f = w.float()

    if use_qk_l2norm_in_kernel:
        q_f = F.normalize(q_f, p=2, dim=-1, eps=1e-6)
        k_f = F.normalize(k_f, p=2, dim=-1, eps=1e-6)

    scale = d_k**-0.5
    q_f = q_f * scale

    if initial_state is not None:
        state = initial_state.float()
    else:
        state = torch.zeros(
            batch_size, num_heads, d_k, d_v, dtype=torch.float32, device=q.device
        )

    outputs = []
    exp_g = torch.exp(g_f)

    for t in range(seq_len):
        q_t = q_f[:, t]
        k_t = k_f[:, t]
        v_t = v_f[:, t]
        g_t = exp_g[:, t]
        b_t = b_f[:, t]
        w_t = w_f[:, t]

        # 1. Decay state along key channels
        state = state * g_t.unsqueeze(-1)

        # 2. Memory read with erase gate
        k_erase = b_t * k_t
        v_read = torch.einsum("bhkv,bhk->bhv", state, k_erase)

        # 3. Delta value with write gate
        v_write = w_t * v_t
        delta = v_write - v_read

        # 4. Update state: S_t = S_{t-1} + k_t delta^T
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta)

        # 5. Output read: o_t = S_t^T q_t
        out_t = torch.einsum("bhkv,bhk->bhv", state, q_t)
        outputs.append(out_t)

    out = torch.stack(outputs, dim=1).to(dtype)
    final_state = state.to(dtype) if output_final_state else None
    return out, final_state


def torch_chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Pure PyTorch chunkwise GDN-2 recurrence.

    Delegates to `torch_recurrent_gdn2` for device-agnostic execution on CPU/non-Triton.
    """
    return torch_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **kwargs,
    )


# Pure PyTorch modules for short convolution and gated RMSNorm


class ShortConvolution(nn.Module):
    """Pure PyTorch depthwise 1D short convolution layer with causal padding."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        bias: bool = False,
        activation: str | None = "silu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t, d = x.shape
        x_t = x.transpose(1, 2)  # [B, D, T]

        if cache is not None and t == 1:
            x_cat = torch.cat([cache, x_t], dim=-1)
            new_cache = x_cat[:, :, 1:] if output_final_state else None
            out = F.conv1d(x_cat, self.conv1d.weight, self.conv1d.bias, groups=d)
        else:
            new_cache = (
                x_t[:, :, -(self.kernel_size - 1) :] if output_final_state else None
            )
            out = F.conv1d(
                x_t,
                self.conv1d.weight,
                self.conv1d.bias,
                padding=self.kernel_size - 1,
                groups=d,
            )[:, :, :t]

        if self.activation == "silu":
            out = F.silu(out)

        out = out.transpose(1, 2)
        return out, new_cache


class RMSNormGated(nn.Module):
    """Pure PyTorch SiLU-gated RMS Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps) * self.weight
        return normed * F.silu(z)


# Hardware-agnostic GatedDeltaNet2 module


class GatedDeltaNet2(nn.Module):
    """Gated DeltaNet 2 (GDN-2) token-mixing layer."""

    def __init__(
        self,
        hidden_size_or_config: int | Any = 2048,
        layer_idx: int | None = None,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: Literal["chunk", "fused_recurrent"] = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Support initialization via config object or explicit parameters
        if hasattr(hidden_size_or_config, "hidden_size") or hasattr(
            hidden_size_or_config, "n_embd"
        ):
            cfg = hidden_size_or_config
            hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 2048))
            num_heads = getattr(cfg, "num_heads", getattr(cfg, "n_head", num_heads))
            head_dim = getattr(
                cfg,
                "head_dim",
                getattr(cfg, "head_size", hidden_size // num_heads),
            )
            num_v_heads = getattr(
                cfg,
                "num_v_heads",
                getattr(cfg, "n_query_groups", num_v_heads or num_heads),
            )
            conv_size = getattr(
                cfg, "conv_size", getattr(cfg, "conv_kernel_size", conv_size)
            )
            norm_eps = getattr(
                cfg, "norm_eps", getattr(cfg, "rms_norm_eps", norm_eps)
            )
            allow_neg_eigval = getattr(
                cfg, "allow_neg_eigval", allow_neg_eigval
            )
            expand_v = getattr(cfg, "expand_v", expand_v)
        else:
            hidden_size = kwargs.pop("hidden_size", int(hidden_size_or_config))

        self.hidden_size = hidden_size
        self.num_heads = kwargs.pop("num_heads", num_heads)
        self.head_k_dim = kwargs.pop("head_dim", head_dim)
        self.num_v_heads = kwargs.pop("num_v_heads", num_v_heads or self.num_heads)
        self.conv_size = kwargs.pop("conv_size", conv_size)
        self.norm_eps = kwargs.pop("norm_eps", norm_eps)
        self.allow_neg_eigval = kwargs.pop("allow_neg_eigval", allow_neg_eigval)
        self.expand_v = kwargs.pop("expand_v", expand_v)

        self.layer_idx = layer_idx
        self.mode = mode
        self.use_short_conv = use_short_conv
        self.conv_bias = conv_bias

        self.head_v_dim = int(self.head_k_dim * self.expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # Short 1D convolutions
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                self.key_dim, kernel_size=self.conv_size, bias=self.conv_bias
            )
            self.k_conv1d = ShortConvolution(
                self.key_dim, kernel_size=self.conv_size, bias=self.conv_bias
            )
            self.v_conv1d = ShortConvolution(
                self.value_dim, kernel_size=self.conv_size, bias=self.conv_bias
            )

        # Decay gate pre-activation projection
        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )

        # Channel-wise erase gate (`b`) and write gate (`w`)
        self.b_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # Output gate projection
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.value_dim, bias=False)
        )

        # Decay-gate parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(0.01, 16.0)
            )
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # Output normalization and projection
        self.o_norm = RMSNormGated(self.head_v_dim, eps=self.norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _get_cache(
        self, past_key_values: Cache | dict[str, Any] | None
    ) -> tuple[
        torch.Tensor | None,
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
    ]:
        if past_key_values is None:
            return None, None

        if isinstance(past_key_values, Cache):
            if (
                hasattr(past_key_values, "has_previous_state")
                and self.layer_idx is not None
                and past_key_values.has_previous_state(self.layer_idx)
            ):
                layers = getattr(past_key_values, "layers", [])
                if self.layer_idx < len(layers):
                    layer_cache = layers[self.layer_idx]
                    rec_state = getattr(layer_cache, "recurrent_states", [None])[0]
                    conv_state = getattr(layer_cache, "conv_states", None)
                    return rec_state, conv_state
            return None, None

        if isinstance(past_key_values, dict):
            return past_key_values.get("recurrent_state"), past_key_values.get(
                "conv_state"
            )

        return None, None

    def _update_cache(
        self,
        past_key_values: Cache | dict[str, Any] | None,
        recurrent_state: torch.Tensor | None,
        conv_state: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
        | None,
    ) -> None:
        if past_key_values is None:
            return

        if self.layer_idx is not None and isinstance(past_key_values, Cache):
            layers = getattr(past_key_values, "layers", [])
            if self.layer_idx < len(layers):
                layer_cache = layers[self.layer_idx]
                is_recurrent_layer = (
                    isinstance(layer_cache, LinearAttentionCacheLayerMixin)
                    or hasattr(layer_cache, "update_recurrent_state")
                    or hasattr(layer_cache, "recurrent_states")
                )
                if (
                    is_recurrent_layer
                    and hasattr(past_key_values, "update_recurrent_state")
                    and recurrent_state is not None
                ):
                    past_key_values.update_recurrent_state(recurrent_state, self.layer_idx)
                if (
                    is_recurrent_layer
                    and hasattr(past_key_values, "update_conv_state")
                    and conv_state is not None
                ):
                    c_state = conv_state[0] if isinstance(conv_state, (tuple, list)) else conv_state
                    if c_state is not None:
                        past_key_values.update_conv_state(c_state, self.layer_idx)
        elif isinstance(past_key_values, dict):
            if recurrent_state is not None:
                past_key_values["recurrent_state"] = recurrent_state
            if conv_state is not None:
                past_key_values["conv_state"] = conv_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | dict[str, Any] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | dict[str, Any] | None]:
        _, q_len, _ = hidden_states.shape
        mode = (
            "fused_recurrent"
            if (q_len <= 64 and not self.training)
            else self.mode
        )

        recurrent_state, conv_states = self._get_cache(past_key_values)
        conv_state_q, conv_state_k, conv_state_v = (
            conv_states if conv_states is not None else (None, None, None)
        )

        cu_seqlens = kwargs.get("cu_seqlens")

        # Convolutions
        if self.use_short_conv:
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache or False,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache or False,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache or False,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # Log decay computation
        g = -self.A_log.float().exp().repeat_interleave(self.head_k_dim) * F.softplus(
            self.f_proj(hidden_states).float() + self.dt_bias.repeat_interleave(self.head_k_dim)
        )

        # Gates
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()

        # Reshape to head dimensions
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k, g, b = (
                repeat(
                    x,
                    "... h d -> ... (h g) d",
                    g=self.num_v_heads // self.num_heads,
                )
                for x in (q, k, g, b)
            )

        if self.allow_neg_eigval:
            b = b * 2.0

        # Dispatch kernel: Triton on CUDA if available, pure PyTorch otherwise
        use_cuda_triton = (
            hidden_states.is_cuda
            and torch.cuda.is_available()
            and _HAS_TRITON_OPS
        )

        o: torch.Tensor | None = None
        if use_cuda_triton:
            if mode == "chunk" and _triton_chunk_gdn2 is not None:
                o, recurrent_state = _triton_chunk_gdn2(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    b=b,
                    w=w,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    initial_state=recurrent_state,
                    output_final_state=use_cache or False,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                    cu_seqlens=cu_seqlens,
                )
            elif _triton_fused_recurrent_gdn2 is not None:
                o, recurrent_state = _triton_fused_recurrent_gdn2(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    b=b,
                    w=w,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    initial_state=recurrent_state,
                    output_final_state=use_cache or False,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                    cu_seqlens=cu_seqlens,
                )
            else:
                use_cuda_triton = False

        if not use_cuda_triton:
            _warn_fallback_once("Triton/CUDA unavailable or CPU tensor")
            if mode == "chunk":
                o, recurrent_state = torch_chunk_gdn2(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    b=b,
                    w=w,
                    initial_state=recurrent_state,
                    output_final_state=use_cache or False,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                o, recurrent_state = torch_recurrent_gdn2(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    b=b,
                    w=w,
                    initial_state=recurrent_state,
                    output_final_state=use_cache or False,
                    use_qk_l2norm_in_kernel=True,
                )

        # Update cache
        if use_cache or past_key_values is not None:
            self._update_cache(
                past_key_values,
                recurrent_state=recurrent_state,
                conv_state=(
                    (conv_state_q, conv_state_k, conv_state_v)
                    if self.use_short_conv
                    else None
                ),
            )

        # Output normalization and projection
        z = rearrange(
            self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
        )
        assert o is not None
        o = self.o_norm(o, z)
        o = rearrange(o, "... h d -> ... (h d)")
        o = self.o_proj(o)

        return o, None, past_key_values
