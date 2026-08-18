# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""GDN-2 (Gated DeltaNet 2) hardware-agnostic token-mixing layer.

This module defines `GatedDeltaNet2`, the `nn.Module` that wraps the GDN-2 recurrence into
a drop-in token mixer for Qwen-style Transformer blocks. It supports both GPU (accelerated
via Triton/FLA when available) and CPU/device-agnostic execution via pure PyTorch reference fallbacks.

GDN-2 extends KDA's scalar-beta erase gate to channel-wise erase (`b`) and write (`w`) gates:
    S_t = (I - k_t (b_t \odot k_t)^T) \text{Diag}(\exp(g_t)) S_{t-1} + k_t (w_t \odot v_t)^T
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, cast

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
    from .gdn2_ops.chunk_gdn2 import _HAS_TRITON_FLA as _CHUNK_HAS_TRITON
    from .gdn2_ops.chunk_gdn2 import chunk_gdn2 as _triton_chunk_gdn2
    from .gdn2_ops.fused_recurrent_gdn2 import _HAS_TRITON_FLA as _RECURRENT_HAS_TRITON
    from .gdn2_ops.fused_recurrent_gdn2 import (
        fused_recurrent_gdn2 as _triton_fused_recurrent_gdn2,
    )

    _HAS_TRITON_OPS = bool(_CHUNK_HAS_TRITON or _RECURRENT_HAS_TRITON)
except (ImportError, AttributeError) as e:
    _triton_chunk_gdn2 = None
    _triton_fused_recurrent_gdn2 = None
    _HAS_TRITON_OPS = False
    _warn_fallback_once(f"Triton ops failed to load: {e}")

GDN2_BACKENDS = (
    "auto",
    "torch",
    "torch-chunk",
    "torch-recurrent",
    "compiled",
    "triton",
    "fla",
)


def resolve_gdn2_backend(
    requested: str,
    *,
    training: bool,
    seq_len: int,
) -> str:
    r"""Resolve the concrete GDN-2 execution backend for a forward call.

    "auto" picks a sensible default: Triton/FLA on CUDA (and the fused recurrent
    path for short inference), pure torch elsewhere (chunk for training/long
    sequences, recurrent for single-token/short inference decode). Forcing any
    other value disables automatic selection entirely.
    """
    if requested not in GDN2_BACKENDS:
        raise ValueError(
            f"Invalid GDN-2 backend '{requested}'. Valid backends: {list(GDN2_BACKENDS)}"
        )
    if requested != "auto":
        return requested

    if torch.cuda.is_available() and _HAS_TRITON_OPS:
        return "triton"
    if not training and seq_len <= 1:
        return "torch-recurrent"
    if training:
        return "torch-chunk"
    if seq_len <= 64:
        return "torch-recurrent"
    return "torch-chunk"


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
        output_final_state: Whether to return the final state.
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to queries and keys.

    Returns:
        A tuple `(out, final_state)` where `out` has shape `[B, T, H, d_v]`.
    """
    batch_size, seq_len, num_heads, d_k = q.shape
    d_v = v.shape[-1]
    dtype = q.dtype

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    b_f = b.float()
    w_f = w.float()

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

    scale = d_k**-0.5
    q = q * scale

    if initial_state is None:
        state = torch.zeros(
            batch_size, num_heads, d_k, d_v, dtype=torch.float32, device=q.device
        )
    else:
        state = initial_state.float()

    outputs = []
    exp_g = torch.exp(g)

    for t in range(seq_len):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        g_t = exp_g[:, t]
        b_t = b_f[:, t]
        w_t = w_f[:, t]

        # 1. Decay state along key channels
        state = state * g_t.unsqueeze(-1)

        # 2. Memory read with erase gate
        erase_k = b_t * k_t
        v_read = torch.einsum("bhkv,bhk->bhv", state, erase_k)

        # 3. Delta value with write gate
        v_write = w_t * v_t - v_read

        # 4. Update state: S_t = S_{t-1} + k_t \delta^T
        state = state + k_t.unsqueeze(-1) * v_write.unsqueeze(-2)

        # 5. Output read: o_t = S_t^T q_t
        out_t = torch.einsum("bhkv,bhk->bhv", state, q_t)
        outputs.append(out_t)

    out = torch.stack(outputs, dim=1).to(dtype)
    final_state = state.to(dtype) if output_final_state else None

    return out, final_state


def compute_gdn2_wy_coefficients(
    kbar: torch.Tensor,
    ebar: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Solve the WY triangular system for one chunk (paper Appendix A).

    Given the decay-normalized key ``kbar = gamma^{-1} * k`` and the
    decay-normalized erase vector ``ebar = gamma * (b * k)`` of one chunk
    (shape ``[B, H, C, K]``) plus the write vector ``z = w * v`` (shape
    ``[B, H, C, V]``), return the WY auxiliary factors

        Y = (I + T)^{-1} ebar      (shape ``[B, H, C, K]``)
        U = (I + T)^{-1} z         (shape ``[B, H, C, V]``)

    where ``T = tril(ebar @ kbar^T, -1)`` is the strictly lower-triangular
    intra-chunk interaction matrix. ``I + T`` is unit lower triangular, so the
    solves are exact, stable, and hardware agnostic.
    """
    c = kbar.shape[-2]
    t_mat = torch.tril(torch.matmul(ebar, kbar.transpose(-1, -2)), diagonal=-1)
    eye = torch.eye(c, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    a_mat = eye + t_mat  # [B, H, C, C] unit lower triangular
    y = torch.linalg.solve_triangular(a_mat, ebar, upper=False, unitriangular=True)
    u = torch.linalg.solve_triangular(a_mat, z, upper=False, unitriangular=True)
    return y, u


def compute_gdn2_intra_chunk_scores(
    q: torch.Tensor,
    gamma: torch.Tensor,
    kbar: torch.Tensor,
) -> torch.Tensor:
    r"""Build the causal intra-chunk output score matrix.

    Args:
        q: Scaled query of shape ``[B, H, C, K]``.
        gamma: Absolute cumulative decay of shape ``[B, H, C, K]``.
        kbar: Decay-normalized key ``gamma^{-1} * k``, shape ``[B, H, C, K]``.

    Returns:
        The causal score matrix ``Aqk`` of shape ``[B, H, C, C]``:

            (Aqk)_{r,i} = 1_{i<=r} (gamma_r * q_r)^T (gamma_i^{-1} * k_i)
                        = 1_{i<=r} q_r^T Diag(gamma_r / gamma_i) k_i

    Entries above the diagonal are zeroed (row ``r`` attends only to ``i <= r``).
    """
    q_gamma = gamma * q  # [B, H, C, K] = Diag(gamma_r) q_r
    scores = torch.matmul(q_gamma, kbar.transpose(-1, -2))  # [B, H, C, C]
    c = scores.shape[-1]
    causal = torch.tril(torch.ones(c, c, device=scores.device, dtype=torch.bool))
    return scores.masked_fill(~causal, 0.0)


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
    chunk_size: int = 64,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Pure PyTorch chunkwise GDN-2 recurrence (paper Appendix A).

    Decay-normalized chunkwise/WY formulation, built only on generic torch ops
    (`matmul`, `solve_triangular`) so it is equivalent to
    `torch_recurrent_gdn2` but parallel across tokens inside each chunk. It is
    the pure-PyTorch correctness reference for the accelerated backends and the
    memory-efficient training path.

    With the absolute cumulative decay ``gamma_r = exp(cumsum(log-decay)_{1..r})``:

        kbar_r = gamma_r^{-1} * k_r        ebar_r = gamma_r * (b_r * k_r)
        z_r = w_r * v_r                    T = tril(ebar @ kbar^T, -1)
        Y = (I+T)^{-1} ebar                U = (I+T)^{-1} z
        delta = U - Y @ S_start            S_next = S_start + kbar^T @ delta
        Aqk = causal_tril(gamma*q @ kbar^T)
        output = gamma*q @ S_start + Aqk @ delta

    Args:
        q: Query tensor ``[B, T, H, d_k]``.
        k: Key tensor ``[B, T, H, d_k]``.
        v: Value tensor ``[B, T, H, d_v]``.
        g: Log-decay tensor ``[B, T, H, d_k]``.
        b: Erase gate tensor ``[B, T, H, d_k]``.
        w: Write gate tensor ``[B, T, H, d_v]``.
        initial_state: Optional recurrent state ``[B, H, d_k, d_v]`` (real-space).
        output_final_state: Whether to return the final state.
        use_qk_l2norm_in_kernel: Whether to L2-normalize q and k.
        chunk_size: Number of tokens processed per chunk.

    Returns:
        ``(out, final_state)`` where ``out`` has shape ``[B, T, H, d_v]``.
    """
    batch_size, seq_len, num_heads, d_k = q.shape
    d_v = v.shape[-1]
    out_dtype = q.dtype

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    b_f = b.float()
    w_f = w.float()

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

    scale = d_k**-0.5
    q = q * scale

    # Move to chunk-friendly layout [B, H, T, D].
    q = rearrange(q, "b t h d -> b h t d")
    k = rearrange(k, "b t h d -> b h t d")
    v = rearrange(v, "b t h d -> b h t d")
    g = rearrange(g, "b t h d -> b h t d")
    b_f = rearrange(b_f, "b t h d -> b h t d")
    w_f = rearrange(w_f, "b t h d -> b h t d")

    if initial_state is None:
        state = torch.zeros(
            batch_size, num_heads, d_k, d_v, dtype=torch.float32, device=q.device
        )
    else:
        # At position 0 the cumulative decay reaches 1, so the normalized state
        # equals the real-space state (matches the reference oracle).
        state = initial_state.float()

    outputs: list[torch.Tensor] = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)

        q_c = q[:, :, start:end]
        k_c = k[:, :, start:end]
        v_c = v[:, :, start:end]
        g_c = g[:, :, start:end]
        b_c = b_f[:, :, start:end]
        w_c = w_f[:, :, start:end]

        # Chunk-local cumulative decay gamma: [B, H, C, K].
        gamma = torch.exp(torch.cumsum(g_c, dim=2))
        gamma_last = gamma[:, :, -1:, :]  # [B, H, 1, K]

        # Decay-normalized factors (paper Eq. 33).
        gam_safe = gamma.clamp_min(1e-12)
        kbar = k_c / gam_safe  # [B, H, C, K]
        ebar = gamma * (b_c * k_c)  # [B, H, C, K]
        z = w_c * v_c  # [B, H, C, V]

        # WY triangular solve.
        y, u = compute_gdn2_wy_coefficients(kbar, ebar, z, device=q.device)

        # Normalized state correction: delta = U - Y @ S_start.
        delta = u - torch.matmul(y, state)  # [B, H, C, V]

        # Output read: out = (gamma*q) @ S_start + Aqk @ delta.
        q_gamma = gamma * q_c  # [B, H, C, K]
        out_inter = torch.matmul(q_gamma, state)  # [B, H, C, V]
        aqk = compute_gdn2_intra_chunk_scores(q_c, gamma, kbar)  # [B, H, C, C]
        out_c = out_inter + torch.matmul(aqk, delta)
        outputs.append(out_c)

        # Carry state to next chunk: S_next = gamma_last^T * (S_start + kbar^T @ delta).
        state = gamma_last.transpose(-1, -2) * (
            state + torch.matmul(kbar.transpose(-1, -2), delta)
        )

    out = torch.cat(outputs, dim=2)  # [B, H, T, V]
    out = rearrange(out, "b h t d -> b t h d").to(out_dtype)

    final_state: torch.Tensor | None = None
    if output_final_state:
        final_state = state.to(out_dtype)

    return out, final_state


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

        if cache is None and output_final_state and t == 1:
            cache = torch.zeros(
                x.shape[0], d, self.kernel_size - 1, device=x.device, dtype=x.dtype
            )

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


class GatedDeltaNet2(nn.Module):
    """Gated DeltaNet 2 (GDN-2) token-mixing layer."""

    def __init__(
        self,
        hidden_size_or_config: int | Any = 2048,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        layer_idx: int | None = None,
        mode: Literal["chunk", "fused_recurrent"] = "chunk",
        expand_v: float = 1.0,
        num_v_heads: int | None = None,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        chunk_size: int = 64,
        backend: str = "auto",
        compile_backend: bool = False,
        fp32_decay: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Support initialization via config object or explicit parameters
        if hasattr(hidden_size_or_config, "hidden_size") or hasattr(
            hidden_size_or_config, "n_embd"
        ):
            cfg = hidden_size_or_config
            hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 2048))
            num_heads = getattr(cfg, "num_heads", getattr(cfg, "n_head", 16))
            head_dim = getattr(cfg, "head_dim", getattr(cfg, "head_size", 128))
            num_v_heads = getattr(
                cfg,
                "num_v_heads",
                getattr(cfg, "n_query_groups", num_v_heads or num_heads),
            )
            conv_size = getattr(
                cfg, "conv_size", getattr(cfg, "conv_kernel_size", conv_size)
            )
            norm_eps = getattr(cfg, "norm_eps", getattr(cfg, "rms_norm_eps", norm_eps))
            allow_neg_eigval = getattr(cfg, "allow_neg_eigval", allow_neg_eigval)
            expand_v = getattr(cfg, "expand_v", expand_v)
            chunk_size = getattr(
                cfg, "chunk_size", getattr(cfg, "train_chunk_size", chunk_size)
            )
            backend = getattr(cfg, "backend", backend)
            compile_backend = getattr(cfg, "compile_backend", compile_backend)
            fp32_decay = getattr(cfg, "fp32_decay", fp32_decay)
        elif hidden_size is None:
            hidden_size = int(hidden_size_or_config)

        if backend not in GDN2_BACKENDS:
            raise ValueError(
                f"Invalid GDN-2 backend '{backend}'. Valid backends: {list(GDN2_BACKENDS)}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads if num_heads is not None else 16
        self.head_dim = head_dim if head_dim is not None else 128
        self.num_v_heads = num_v_heads if num_v_heads is not None else self.num_heads
        self.layer_idx = layer_idx
        self.mode = mode
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.norm_eps = norm_eps
        self.expand_v = expand_v
        self.chunk_size = chunk_size
        self.backend = backend
        self.compile_backend = compile_backend
        self.fp32_decay = fp32_decay

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)

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

        # Decay-gate parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(0.1, 2.0)
            )
        )
        cast(Any, self.A_log)._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        cast(Any, self.dt_bias)._no_weight_decay = True

        # Output normalization and projection
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNormGated(self.head_v_dim, eps=self.norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.apply(self._initialize_weights)

        # Optional torch.compile backend. Best-effort: if induction fails (e.g.
        # generic non-CPU/cuda devices), we transparently fall back to the plain
        # chunk kernel so the signature stays identical.
        self._compiled_chunk: Any = None
        if self.compile_backend:
            try:
                self._compiled_chunk = torch.compile(torch_chunk_gdn2, dynamic=True)
            except (
                RuntimeError,
                ValueError,
                TypeError,
                AttributeError,
            ) as e:  # compile is purely optional
                _warn_fallback_once(f"torch.compile unavailable ({e})")
                self._compiled_chunk = None

    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNormGated):
            nn.init.ones_(module.weight)
        cast(Any, module)._is_hf_initialized = True

    def _get_cache(
        self, past_key_values: Cache | dict[str, Any] | None
    ) -> tuple[
        torch.Tensor | None,
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
    ]:
        if past_key_values is None:
            return None, None

        if isinstance(past_key_values, Cache):
            layers = getattr(past_key_values, "layers", [])
            if self.layer_idx is not None and self.layer_idx < len(layers):
                layer_cache = layers[self.layer_idx]

                rec_states = getattr(layer_cache, "recurrent_states", None)
                if rec_states is None:
                    rec_state = getattr(layer_cache, "recurrent_state", None)
                elif isinstance(rec_states, torch.Tensor):
                    rec_state = rec_states
                elif isinstance(rec_states, dict):
                    rec_state = rec_states.get(0)
                elif isinstance(rec_states, (list, tuple)) and len(rec_states) > 0:
                    rec_state = rec_states[0]
                else:
                    rec_state = None

                conv_states = getattr(layer_cache, "conv_states", None)
                if conv_states is None:
                    conv_state = getattr(layer_cache, "conv_state", None)
                elif isinstance(conv_states, dict):
                    conv_state = (
                        conv_states.get(0),
                        conv_states.get(1),
                        conv_states.get(2),
                    )
                elif isinstance(conv_states, (list, tuple)) and len(conv_states) == 3:
                    conv_state = (conv_states[0], conv_states[1], conv_states[2])
                else:
                    conv_state = None

                return rec_state, conv_state
            return None, None

        if isinstance(past_key_values, dict):
            rec = past_key_values.get("recurrent_state")
            conv = past_key_values.get("conv_state")
            return rec, conv

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
                    try:
                        past_key_values.update_recurrent_state(
                            recurrent_state, self.layer_idx
                        )
                    except (
                        TypeError,
                        ValueError,
                        AttributeError,
                        RuntimeError,
                        IndexError,
                    ) as e:
                        _warn_fallback_once(f"update_recurrent_state failed: {e}")
                elif recurrent_state is not None:
                    rec_dict = getattr(layer_cache, "recurrent_states", None)
                    if isinstance(rec_dict, dict):
                        rec_dict[0] = recurrent_state
                    elif hasattr(layer_cache, "recurrent_state"):
                        layer_cache.recurrent_state = recurrent_state

                if (
                    is_recurrent_layer
                    and hasattr(past_key_values, "update_conv_state")
                    and conv_state is not None
                ):
                    try:
                        past_key_values.update_conv_state(
                            cast(Any, conv_state), self.layer_idx
                        )
                    except (
                        TypeError,
                        ValueError,
                        AttributeError,
                        RuntimeError,
                        IndexError,
                    ) as e:
                        _warn_fallback_once(f"update_conv_state failed: {e}")
                elif conv_state is not None:
                    conv_dict = getattr(layer_cache, "conv_states", None)
                    if isinstance(conv_dict, dict):
                        conv_dict[0] = conv_state[0]
                        conv_dict[1] = conv_state[1]
                        conv_dict[2] = conv_state[2]
                    elif hasattr(layer_cache, "conv_state"):
                        layer_cache.conv_state = conv_state

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
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode

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

        g = -self.A_log.float().exp().repeat_interleave(self.head_k_dim) * F.softplus(
            self.f_proj(hidden_states).float() + self.dt_bias
        )
        # The decay activation is always computed in fp32 (paper Sec. D.1); toggle
        # whether it stays fp32 end-to-end or is cast to the model dtype.
        g = g.float() if self.fp32_decay else g.to(hidden_states.dtype)

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
            groups = self.num_v_heads // self.num_heads
            q, k, g, b = (
                repeat(x, "... h d -> ... (h g) d", g=groups) for x in (q, k, g, b)
            )

        if self.allow_neg_eigval:
            b = b * 2.0

        # Resolve and dispatch the concrete backend. The GDN-2 maths below never
        # branches on device; only this block performs the (auto) selection.
        backend = resolve_gdn2_backend(
            self.backend, training=self.training, seq_len=q_len
        )
        o: torch.Tensor | None = None

        if backend in ("triton", "fla"):
            if not _HAS_TRITON_OPS:
                raise RuntimeError(
                    f"GDN-2 backend '{backend}' was requested but Triton/FLA is not installed. "
                    "Install the optional CUDA dependencies (pip install 'qwendopamine[gpu]') "
                    "or use backend='auto'/'torch'."
                )
            try:
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
                    raise RuntimeError("Triton ops present but no kernel is callable")
            except (
                RuntimeError,
                TypeError,
                ValueError,
                AttributeError,
                ImportError,
            ) as e:
                _warn_fallback_once(
                    f"Triton kernel failed ({e}); falling back to pure PyTorch"
                )
                backend = "torch-chunk" if mode == "chunk" else "torch-recurrent"

        if o is None:
            # Pure PyTorch / compiled path. `torch` follows the selected mode;
            # `torch-chunk` and `compiled` always chunk; `torch-recurrent` always
            # runs the token loop.
            chunk_fn = torch_chunk_gdn2
            if backend == "compiled" and self._compiled_chunk is not None:
                chunk_fn = self._compiled_chunk

            def _chunk_call() -> tuple[torch.Tensor, torch.Tensor | None]:
                return chunk_fn(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    b=b,
                    w=w,
                    initial_state=recurrent_state,
                    output_final_state=use_cache or False,
                    use_qk_l2norm_in_kernel=True,
                    chunk_size=self.chunk_size,
                )

            if backend in ("torch-chunk", "compiled"):
                o, recurrent_state = _chunk_call()
            elif backend == "torch-recurrent":
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
            else:  # backend == "torch": respect the selected mode
                if mode == "fused_recurrent":
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
                else:
                    o, recurrent_state = _chunk_call()

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
        gate = rearrange(
            self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
        )
        assert o is not None
        o = self.o_norm(o, gate)
        o = rearrange(o, "... h d -> ... (h d)")
        out = self.o_proj(o)

        return out, None, past_key_values
