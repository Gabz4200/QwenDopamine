# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Gated DeltaNet 2 (GDN-2) token-mixing block.

This module defines :class:`GatedDeltaNet2`, the ``nn.Module`` that wraps the
GDN-2 recurrence into a drop-in token mixer for Qwen-style Transformer blocks.
It supports both GPU (accelerated via Triton/FLA when available) and CPU/
device-agnostic execution via pure PyTorch reference fallbacks.

GDN-2 extends KDA's scalar-beta erase gate to channel-wise erase (``b``) and
write (``w``) gates:

    S_t = (I - k_t (b_t \odot k_t)^T) \text{Diag}(\exp(g_t)) S_{t-1} + k_t (w_t \odot v_t)^T

The implementation is decomposed into focused helper methods for projections,
convolution pre-filtering, backend dispatch, and cache management.
"""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from transformers.cache_utils import Cache

from qwendopamine.models.gdn2.backend import resolve_gdn2_backend
from qwendopamine.models.gdn2.cache_utils import (
    get_unpad_data,
    index_first_axis,
    pad_input,
)
from qwendopamine.models.gdn2.chunk import torch_chunk_gdn2
from qwendopamine.models.gdn2.convolution import ShortConvolution
from qwendopamine.models.gdn2.core import torch_recurrent_gdn2
from qwendopamine.models.gdn2.normalization import RMSNormGated

try:
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
except ImportError:
    LinearAttentionCacheLayerMixin = type(None)  # type: ignore[misc, assignment]

# Safe optional Triton/FLA ops imports
_HAS_TRITON_OPS = False
try:
    from qwendopamine.models.gdn2.gdn2_ops.chunk_gdn2 import (
        _HAS_TRITON_FLA as _CHUNK_HAS_TRITON,
    )
    from qwendopamine.models.gdn2.gdn2_ops.chunk_gdn2 import (
        chunk_gdn2 as _triton_chunk_gdn2,
    )
    from qwendopamine.models.gdn2.gdn2_ops.fused_recurrent_gdn2 import (
        _HAS_TRITON_FLA as _RECURRENT_HAS_TRITON,
    )
    from qwendopamine.models.gdn2.gdn2_ops.fused_recurrent_gdn2 import (
        fused_recurrent_gdn2 as _triton_fused_recurrent_gdn2,
    )

    _HAS_TRITON_OPS = bool(_CHUNK_HAS_TRITON or _RECURRENT_HAS_TRITON)
except (ImportError, AttributeError) as e:
    _triton_chunk_gdn2 = None
    _triton_fused_recurrent_gdn2 = None
    _HAS_TRITON_OPS = False
    from qwendopamine.models.gdn2.backend import _warn_fallback_once

    _warn_fallback_once(f"Triton ops failed to load: {e}")

_DEFAULT_HIDDEN_SIZE = 2048
_DEFAULT_NUM_HEADS = 16
_DEFAULT_HEAD_DIM = 128
_DEFAULT_MODE = "chunk"
_DEFAULT_EXPAND_V = 1.0
_DEFAULT_USE_SHORT_CONV = True
_DEFAULT_ALLOW_NEG_EIGVAL = False
_DEFAULT_CONV_SIZE = 4
_DEFAULT_CONV_BIAS = False
_DEFAULT_NORM_EPS = 1e-5
_DEFAULT_CHUNK_SIZE = 64
_DEFAULT_BACKEND = "auto"
_DEFAULT_COMPILE_BACKEND = False
_DEFAULT_FP32_DECAY = True


class GatedDeltaNet2(nn.Module):
    """Gated DeltaNet 2 (GDN-2) token-mixing layer."""

    def __init__(
        self,
        hidden_size_or_config: int | Any = _DEFAULT_HIDDEN_SIZE,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        layer_idx: int | None = None,
        mode: Literal["chunk", "fused_recurrent"] = _DEFAULT_MODE,
        expand_v: float = _DEFAULT_EXPAND_V,
        num_v_heads: int | None = None,
        use_short_conv: bool = _DEFAULT_USE_SHORT_CONV,
        allow_neg_eigval: bool = _DEFAULT_ALLOW_NEG_EIGVAL,
        conv_size: int = _DEFAULT_CONV_SIZE,
        conv_bias: bool = _DEFAULT_CONV_BIAS,
        norm_eps: float = _DEFAULT_NORM_EPS,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        backend: str = _DEFAULT_BACKEND,
        compile_backend: bool = _DEFAULT_COMPILE_BACKEND,
        fp32_decay: bool = _DEFAULT_FP32_DECAY,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Support initialization via config object or explicit parameters
        if hasattr(hidden_size_or_config, "hidden_size") or hasattr(
            hidden_size_or_config, "n_embd"
        ):
            cfg = hidden_size_or_config
            hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", _DEFAULT_HIDDEN_SIZE))
            num_heads = getattr(cfg, "num_heads", getattr(cfg, "n_head", _DEFAULT_NUM_HEADS))
            head_dim = getattr(cfg, "head_dim", getattr(cfg, "head_size", _DEFAULT_HEAD_DIM))
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

        if backend not in _GATED_DELTA_NET_BACKENDS:
            raise ValueError(
                f"Invalid GDN-2 backend '{backend}'. Valid backends: {list(_GATED_DELTA_NET_BACKENDS)}"
            )

        self._init_hyperparameters(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_v_heads=num_v_heads,
            layer_idx=layer_idx,
            mode=mode,
            expand_v=expand_v,
            use_short_conv=use_short_conv,
            allow_neg_eigval=allow_neg_eigval,
            conv_size=conv_size,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            chunk_size=chunk_size,
            backend=backend,
            compile_backend=compile_backend,
            fp32_decay=fp32_decay,
        )
        self._init_projections()
        self._init_decay_parameters()
        self._init_output_head()
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

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init_hyperparameters(self, **kwargs: Any) -> None:
        (self.hidden_size,) = (kwargs["hidden_size"],)
        self.num_heads = kwargs["num_heads"] or _DEFAULT_NUM_HEADS
        self.head_dim = kwargs["head_dim"] or _DEFAULT_HEAD_DIM
        self.num_v_heads = kwargs["num_v_heads"] or self.num_heads
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )
        self.layer_idx = kwargs["layer_idx"]
        self.mode = kwargs["mode"]
        self.use_short_conv = kwargs["use_short_conv"]
        self.allow_neg_eigval = kwargs["allow_neg_eigval"]
        self.conv_size = kwargs["conv_size"]
        self.conv_bias = kwargs["conv_bias"]
        self.norm_eps = kwargs["norm_eps"]
        self.expand_v = kwargs["expand_v"]
        self.chunk_size = kwargs["chunk_size"]
        self.backend = kwargs["backend"]
        self.compile_backend = kwargs["compile_backend"]
        self.fp32_decay = kwargs["fp32_decay"]

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)

    def _init_projections(self) -> None:
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

        self.b_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

    def _init_decay_parameters(self) -> None:
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
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

    def _init_output_head(self) -> None:
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNormGated(self.head_v_dim, eps=self.norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

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

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------
    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Initialize the O(1) recurrent memory matrix S_0.

        Args:
            batch_size: Number of sequences in the batch.
            device: Target device for the state tensor.
            dtype: Target dtype for the state tensor.

        Returns:
            Zero-initialized state of shape ``[B, H, K, V]``.
        """
        return torch.zeros(
            batch_size, self.num_heads, self.head_k_dim, self.head_v_dim,
            device=device, dtype=dtype
        )

    # ------------------------------------------------------------------
    # Single-token auto-regressive step
    # ------------------------------------------------------------------
    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | None = None,
        conv_state_q: torch.Tensor | None = None,
        conv_state_k: torch.Tensor | None = None,
        conv_state_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
        """Execute a single auto-regressive token step on CPU/GPU.

        This is the preferred API for inference-time decoding. It updates the
        matrix-valued memory state in-place without storing historical token
        representations, keeping the cache size O(1) per layer.

        Args:
            x_t: Input tensor for a single step of shape ``[B, 1, D]`` or ``[B, D]``.
            state: Hidden memory state S_{t-1} of shape ``[B, H, K, V]``.
            conv_state_q: Conv cache for Q of shape ``[B, D, kernel_size-1]``.
            conv_state_k: Conv cache for K of shape ``[B, D, kernel_size-1]``.
            conv_state_v: Conv cache for V of shape ``[B, D, kernel_size-1]``.

        Returns:
            out_t: Output tensor for the step of shape ``[B, 1, D]``.
            new_state: Updated hidden memory state S_t of shape ``[B, H, K, V]``.
            new_conv_states: Tuple of updated conv caches ``(q, k, v)``.
        """
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(1)

        B = x_t.shape[0]
        H, K = self.num_heads, self.head_k_dim
        V = self.head_v_dim

        if state is None:
            state = self.init_state(B, x_t.device, x_t.dtype)

        # Project single token vectors
        if self.use_short_conv:
            q, new_q_conv = self.q_conv1d(
                self.q_proj(x_t), cache=conv_state_q, output_final_state=True
            )
            k, new_k_conv = self.k_conv1d(
                self.k_proj(x_t), cache=conv_state_k, output_final_state=True
            )
            v, new_v_conv = self.v_conv1d(
                self.v_proj(x_t), cache=conv_state_v, output_final_state=True
            )
        else:
            q = F.silu(self.q_proj(x_t))
            k = F.silu(self.k_proj(x_t))
            v = F.silu(self.v_proj(x_t))
            new_q_conv = new_k_conv = new_v_conv = None

        q = F.normalize(q.view(B, 1, H, K), p=2, dim=-1).view(B, H, K)
        k = F.normalize(k.view(B, 1, H, K), p=2, dim=-1).view(B, H, K)
        v = v.view(B, 1, H, V).view(B, H, V)

        g = -self.A_log.float().exp().repeat_interleave(K) * F.softplus(
            self.f_proj(x_t).float() + self.dt_bias
        )
        g = g.float() if self.fp32_decay else g.to(x_t.dtype)
        g = g.view(B, 1, H, K).view(B, H, K)

        b = self.b_proj(x_t).sigmoid().view(B, 1, H, K).view(B, H, K)
        w = self.w_proj(x_t).sigmoid().view(B, 1, H, V).view(B, H, V)

        # Single-step recurrence via the core engine
        y_t, new_state = torch_recurrent_gdn2(
            q=q.unsqueeze(1),
            k=k.unsqueeze(1),
            v=v.unsqueeze(1),
            g=g.unsqueeze(1),
            b=b.unsqueeze(1),
            w=w.unsqueeze(1),
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        # new_state is [B, H, K, V] from torch_recurrent_gdn2
        y_t = y_t.squeeze(1)  # [B, H, V]
        assert new_state is not None
        new_state = cast(torch.Tensor, new_state)
        y_t = cast(torch.Tensor, y_t)

        # Output gate
        gate = self.g_proj(x_t).view(B, 1, H, V).view(B, H, V)
        out_t = self.o_norm(y_t, gate)  # [B, H, V]
        out_t = self.o_proj(out_t.reshape(B, 1, H * V))  # [B, 1, D]

        return out_t, new_state, (new_q_conv, new_k_conv, new_v_conv)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
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
                        from qwendopamine.models.gdn2.backend import _warn_fallback_once
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
                        from qwendopamine.models.gdn2.backend import _warn_fallback_once
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

    # ------------------------------------------------------------------
    # QKV projections and gate computation
    # ------------------------------------------------------------------
    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        conv_state_q: torch.Tensor | None,
        conv_state_k: torch.Tensor | None,
        conv_state_v: torch.Tensor | None,
        use_cache: bool,
        cu_seqlens: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        r"""Compute q, k, v projections and decay gates from hidden states."""
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
        g = g.float() if self.fp32_decay else g.to(hidden_states.dtype)

        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()
        return q, k, v, g, b, w, conv_state_q, conv_state_k, conv_state_v

    def _prepare_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Rearrange projections into per-head layout and apply value-head grouping."""
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
        return q, k, v, g, b, w

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------
    def _dispatch_backend(
        self,
        backend: str,
        mode: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        recurrent_state: torch.Tensor | None,
        use_cache: bool,
        cu_seqlens: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Dispatch the forward pass to the selected GDN-2 backend."""
        if backend in ("triton", "fla"):
            return self._run_triton_backend(
                backend, mode, q, k, v, g, b, w, recurrent_state, use_cache, cu_seqlens
            )
        return self._run_torch_backend(
            backend, mode, q, k, v, g, b, w, recurrent_state, use_cache
        )

    def _run_triton_backend(
        self,
        backend: str,
        mode: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        recurrent_state: torch.Tensor | None,
        use_cache: bool,
        cu_seqlens: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Run the Triton/FLA backend with automatic fallback to pure PyTorch."""
        if not _HAS_TRITON_OPS:
            raise RuntimeError(
                f"GDN-2 backend '{backend}' was requested but Triton/FLA is not installed. "
                "Install the optional CUDA dependencies (pip install 'qwendopamine[gpu]') "
                "or use backend='auto'/'torch'."
            )
        try:
            if mode == "chunk" and _triton_chunk_gdn2 is not None:
                return _triton_chunk_gdn2(
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
            if _triton_fused_recurrent_gdn2 is not None:
                return _triton_fused_recurrent_gdn2(
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
            raise RuntimeError("Triton ops present but no kernel is callable")
        except (
            RuntimeError,
            TypeError,
            ValueError,
            AttributeError,
            ImportError,
        ) as e:
            from qwendopamine.models.gdn2.backend import _warn_fallback_once
            _warn_fallback_once(
                f"Triton kernel failed ({e}); falling back to pure PyTorch"
            )
            fallback = "torch-chunk" if mode == "chunk" else "torch-recurrent"
            return self._run_torch_backend(
                fallback, mode, q, k, v, g, b, w, recurrent_state, use_cache
            )

    def _run_torch_backend(
        self,
        backend: str,
        mode: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        recurrent_state: torch.Tensor | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Run the pure-PyTorch or compiled backend."""
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
            return _chunk_call()
        if backend == "torch-recurrent":
            return torch_recurrent_gdn2(
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
        # backend == "torch": respect the selected mode
        if mode == "fused_recurrent":
            return torch_recurrent_gdn2(
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
        return _chunk_call()

    # ------------------------------------------------------------------
    # Output projection
    # ------------------------------------------------------------------
    def _compute_output(
        self,
        hidden_states: torch.Tensor,
        o: torch.Tensor,
        g: torch.Tensor,
        is_padded: bool,
        indices: torch.Tensor | None,
        batch: int,
        seq_len: int,
    ) -> torch.Tensor:
        r"""Apply the output gate, normalization, projection, and optional unpadding."""
        gate = rearrange(
            self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
        )
        if o is None:
            raise RuntimeError("GDN-2 backend returned None output.")
        o = self.o_norm(o, gate)
        o = rearrange(o, "... h d -> ... (h d)")
        out = self.o_proj(o)
        if is_padded and indices is not None:
            out = pad_input(out.squeeze(0), indices, batch, seq_len)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | dict[str, Any] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | dict[str, Any] | None]:
        batch, seq_len, _ = hidden_states.shape
        mode = "fused_recurrent" if (seq_len <= 64 and not self.training) else self.mode

        recurrent_state, conv_states = self._get_cache(past_key_values)
        conv_state_q, conv_state_k, conv_state_v = (
            conv_states if conv_states is not None else (None, None, None)
        )

        cu_seqlens = kwargs.get("cu_seqlens")

        # Padding masking via unpad/repad, mirroring the NVlabs reference: tokens
        # with `attention_mask == 0` are dropped, the packed sequence runs through
        # the layer, and the output is scattered back to the padded layout. Skipped
        # for single-token decode and whenever a cache is already live, where the
        # batch layout must be preserved for the recurrent state.
        is_padded = (
            attention_mask is not None
            and seq_len > 1
            and past_key_values is None
            and bool((attention_mask == 0).any())
        )
        indices: torch.Tensor | None = None
        if is_padded:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask)
            hidden_states = index_first_axis(hidden_states, indices).unsqueeze(0)

        q, k, v, g, b, w, conv_state_q, conv_state_k, conv_state_v = self._compute_qkv(
            hidden_states, conv_state_q, conv_state_k, conv_state_v,
            use_cache if use_cache is not None else False, cu_seqlens
        )
        q, k, v, g, b, w = self._prepare_tokens(q, k, v, g, b, w)

        backend = resolve_gdn2_backend(
            self.backend, training=self.training, seq_len=seq_len
        )
        o, recurrent_state = self._dispatch_backend(
            backend, mode, q, k, v, g, b, w, recurrent_state,
            use_cache if use_cache is not None else False, cu_seqlens
        )

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

        return self._compute_output(hidden_states, o, g, is_padded, indices, batch, seq_len), None, past_key_values


# Backends constant accessible from block module
_GATED_DELTA_NET_BACKENDS = (
    "auto",
    "torch",
    "torch-chunk",
    "torch-recurrent",
    "compiled",
    "triton",
    "fla",
)


__all__ = ["GatedDeltaNet2"]
