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
from torch import nn
from transformers.cache_utils import Cache

from qwendopamine.models.gdn2.backend import resolve_gdn2_backend
from qwendopamine.models.gdn2.ops.conv import ShortConvolution
from qwendopamine.models.gdn2.ops.norm import RMSNormGated
from qwendopamine.models.gdn2.recurrence.chunk import torch_chunk_gdn2
from qwendopamine.models.gdn2.recurrence.packing import (
    get_unpad_data,
    index_first_axis,
)
from qwendopamine.models.gdn2.recurrence.recurrent import torch_recurrent_gdn2

try:
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
except ImportError:
    LinearAttentionCacheLayerMixin = type(None)  # type: ignore[misc, assignment]

# Safe optional Taichi ops imports via the public ops layer. The Taichi
# backend is the single hardware-accelerated path; it JIT-compiles to
# native CPU code on CPU and to GPU shaders on CUDA, so no separate
# CUDA dependency is required. The model layer never imports the
# Taichi kernels directly — it asks the ops layer to dispatch.
_HAS_TAICHI_OPS = False
_taichi_chunk_gdn2 = None
_taichi_recurrent_gdn2 = None
try:
    from qwendopamine.kernels.taichi import is_available as _taichi_is_available
    from qwendopamine.kernels.taichi.gdn2_api import (
        chunk_taichi_gdn2 as _taichi_chunk_gdn2,
    )
    from qwendopamine.kernels.taichi.gdn2_api import (
        recurrent_taichi_gdn2 as _taichi_recurrent_gdn2,
    )

    _HAS_TAICHI_OPS = bool(_taichi_is_available())
except (ImportError, RuntimeError) as e:
    from qwendopamine.models.gdn2.backend import _warn_fallback_once

    _warn_fallback_once(f"Taichi ops failed to load: {e}")

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
    r"""Gated DeltaNet 2 (GDN-2) token-mixing layer.

    Implements the GDN-2 recurrence (paper Eq. 10), extending KDA's scalar-beta
    erase gate to channel-wise erase (``b``) and write (``w``) gates:

    .. math::

        S_t = (I - k_t (b_t \odot k_t)^T) \text{Diag}(\exp(g_t)) S_{t-1}
              + k_t (w_t \odot v_t)^T

    The implementation is decomposed into focused helper methods for
    projections, backend dispatch, and cache management.

    Shapes:
        S : ``[B, H, K, V]`` — recurrent memory state.
        q : ``[B, T, H, K]`` — query projections.
        k : ``[B, T, H, K]`` — key projections.
        v : ``[B, T, H, V]`` — value projections.
        b : ``[B, T, H, K]`` — channel-wise erase gate.
        w : ``[B, T, H, V]`` — channel-wise write gate.
        g : ``[B, T, H, K]`` — log-decay gate (``a_t = \exp(g_t)``).

    Returns:
        y : ``[B, T, H, V]`` — mixed output.
        S : ``[B, H, K, V]`` — updated recurrent state.

    Dtype contract:
        The recurrent state ``S`` is **always** carried in the caller's
        input dtype; the Taichi kernel and the torch reference both
        upcast internally to float32 for accumulation and cast back on
        return. The kernel's internal dtype is therefore decoupled
        from ``S``'s external dtype — ``S`` is ``state.dtype`` on every
        public boundary.

    GQA support:
        ``num_v_heads`` is exposed in the constructor and may differ
        from ``num_heads`` to enable grouped-query attention. When
        ``num_v_heads > num_heads`` the ``Q``, ``K``, ``b`` and ``g``
        projections are repeated per value-head group; ``V`` and
        ``w`` remain per-value-head. The check
        ``num_v_heads % num_heads == 0`` runs at construction.
    """

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
        r"""__init__(hidden_size_or_config=2048, hidden_size=None, num_heads=None, head_dim=None, layer_idx=None, mode="chunk", expand_v=1.0, num_v_heads=None, use_short_conv=True, allow_neg_eigval=False, conv_size=4, conv_bias=False, norm_eps=1e-5, chunk_size=64, backend="auto", compile_backend=False, fp32_decay=True, **kwargs) -> None

        Initialize the GDN-2 token-mixing layer.

        Accepts either an explicit field-by-field invocation or a single
        config object as the first positional argument.

        Args:
            hidden_size_or_config (int | Any): First positional. A config
                object (anything with ``hidden_size`` or ``n_embd``) or an int.
            hidden_size (int | None): Hidden dimension. Default: ``None``.
            num_heads (int | None): Number of query heads. Default: ``None``.
            head_dim (int | None): Per-head dimension. Default: ``None``.
            layer_idx (int | None): Layer index. Default: ``None``.
            mode (Literal): ``"chunk"`` or ``"fused_recurrent"``. Default: ``"chunk"``.
            expand_v (float): Value head expansion. Default: ``1.0``.
            num_v_heads (int | None): Number of value heads. Default: ``None``.
            use_short_conv (bool): Use the short-conv pre-filter. Default: ``True``.
            allow_neg_eigval (bool): Allow negative eigenvalues. Default: ``False``.
            conv_size (int): Conv kernel size. Default: ``4``.
            conv_bias (bool): Conv bias. Default: ``False``.
            norm_eps (float): RMS norm epsilon. Default: ``1e-5``.
            chunk_size (int): Chunk size. Default: ``64``.
            backend (str): Backend identifier. Default: ``"auto"``.
            compile_backend (bool): Use ``torch.compile``. Default: ``False``.
            fp32_decay (bool): Upcast decay to float32. Default: ``True``.
            **kwargs: Extra fields forwarded to ``build_init_kwargs``.
        """
        super().__init__()

        from qwendopamine.models.gdn2._init import build_init_kwargs

        resolved = build_init_kwargs(
            hidden_size_or_config=hidden_size_or_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            mode=mode,
            expand_v=expand_v,
            num_v_heads=num_v_heads,
            use_short_conv=use_short_conv,
            allow_neg_eigval=allow_neg_eigval,
            conv_size=conv_size,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            chunk_size=chunk_size,
            backend=backend,
            compile_backend=compile_backend,
            fp32_decay=fp32_decay,
            **kwargs,
        )
        self._init_hyperparameters(**resolved)
        self._init_projections()
        self._init_decay_parameters()
        self._init_output_head()
        self.apply(self._initialize_weights)

        # Optional torch.compile backend. Best-effort: if induction fails (e.g.
        # generic non-CPU/cuda devices), we transparently fall back to the plain
        # chunk kernel so the signature stays identical. The exception is
        # captured on ``self.compile_error`` so the user can introspect why
        # the compile failed without grepping logs.
        self._compiled_chunk: Any = None
        self.compile_error: BaseException | None = None
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
                self.compile_error = e

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
        from ._init_guard import is_already_initialised, mark_initialised

        if is_already_initialised(module):
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
        mark_initialised(module)

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Initialise the O(1) recurrent memory matrix ``S_0``.

        Experimental API for autoregressive decoding outside the
        ``forward`` path (greedy generation, beam search, custom
        cache implementations). The production training/inference
        path uses :meth:`forward` and never calls this directly.

        Args:
            batch_size: Number of sequences in the batch.
            device: Target device for the state tensor.
            dtype: Target dtype for the state tensor.

        Returns:
            Zero-initialized state of shape ``[B, H, K, V]``.
        """
        return torch.zeros(
            batch_size,
            self.num_heads,
            self.head_k_dim,
            self.head_v_dim,
            device=device,
            dtype=dtype,
        )

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | None = None,
        conv_state_q: torch.Tensor | None = None,
        conv_state_k: torch.Tensor | None = None,
        conv_state_v: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    ]:
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

    def _get_cache(
        self, past_key_values: Cache | dict[str, Any] | None
    ) -> tuple[
        torch.Tensor | None,
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
    ]:
        r"""_get_cache(past_key_values: Cache | dict[str, Any] | None) -> tuple[torch.Tensor | None, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None]

        Delegate to :func:`qwendopamine.models.gdn2._cache.get_cache`.

        Args:
            past_key_values (Cache | dict[str, Any] | None): The cache to read.

        Returns:
            tuple[torch.Tensor | None, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None]:
            ``(recurrent_state, conv_state)``.
        """
        from qwendopamine.models.gdn2._cache import get_cache

        return get_cache(self.layer_idx, past_key_values)

    def _update_cache(
        self,
        past_key_values: Cache | dict[str, Any] | None,
        recurrent_state: torch.Tensor | None,
        conv_state: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
        | None,
    ) -> None:
        r"""_update_cache(past_key_values, recurrent_state, conv_state) -> None

        Delegate to :func:`qwendopamine.models.gdn2._cache.update_cache`.

        Args:
            past_key_values (Cache | dict[str, Any] | None): The cache to update.
            recurrent_state (torch.Tensor | None): Updated recurrent state.
            conv_state (tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None):
                Updated short-conv state.
        """
        from qwendopamine.models.gdn2._cache import update_cache

        update_cache(self.layer_idx, past_key_values, recurrent_state, conv_state)

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
        r"""_compute_qkv(hidden_states, conv_state_q, conv_state_k, conv_state_v, use_cache, cu_seqlens) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]

        Delegate to :func:`qwendopamine.models.gdn2._qkv.compute_qkv`.

        Args:
            hidden_states (torch.Tensor): Input ``[B, T, D]``.
            conv_state_q (torch.Tensor | None): Conv cache for Q.
            conv_state_k (torch.Tensor | None): Conv cache for K.
            conv_state_v (torch.Tensor | None): Conv cache for V.
            use_cache (bool): Whether to return updated conv states.
            cu_seqlens (Any): Cumulative sequence lengths for variable-length input.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            ``(q, k, v, g, b, w, conv_state_q, conv_state_k, conv_state_v)``.
        """
        from qwendopamine.models.gdn2._qkv import compute_qkv

        return compute_qkv(
            hidden_states,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.f_proj,
            self.b_proj,
            self.w_proj,
            self.q_conv1d if self.use_short_conv else None,
            self.k_conv1d if self.use_short_conv else None,
            self.v_conv1d if self.use_short_conv else None,
            self.use_short_conv,
            conv_state_q,
            conv_state_k,
            conv_state_v,
            self.A_log,
            self.dt_bias,
            self.head_k_dim,
            self.fp32_decay,
            use_cache,
            cu_seqlens,
        )

    def _prepare_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        r"""_prepare_tokens(q, k, v, g, b, w) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

        Delegate to :func:`qwendopamine.models.gdn2._prepare.prepare_tokens`.

        Args:
            q (torch.Tensor): Query ``[B, T, H, K]``.
            k (torch.Tensor): Key ``[B, T, H, K]``.
            v (torch.Tensor): Value ``[B, T, H, V]``.
            g (torch.Tensor): Decay ``[B, T, H, K]``.
            b (torch.Tensor): Erase gate ``[B, T, H, K]``.
            w (torch.Tensor): Write gate ``[B, T, H, V]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            ``(q, k, v, g, b, w)`` in per-head layout.
        """
        from qwendopamine.models.gdn2._prepare import prepare_tokens

        return prepare_tokens(
            q,
            k,
            v,
            g,
            b,
            w,
            self.head_v_dim,
            self.head_k_dim,
            self.num_heads,
            self.num_v_heads,
            self.allow_neg_eigval,
        )

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
        r"""_dispatch_backend(backend, mode, q, k, v, g, b, w, recurrent_state, use_cache, cu_seqlens) -> tuple[torch.Tensor, torch.Tensor | None]

        Delegate to :func:`qwendopamine.models.gdn2._prepare.dispatch_backend`.

        Args:
            backend (str): Selected backend identifier.
            mode (str): Selected mode.
            q (torch.Tensor): Query ``[B, T, H, K]``.
            k (torch.Tensor): Key ``[B, T, H, K]``.
            v (torch.Tensor): Value ``[B, T, H, V]``.
            g (torch.Tensor): Decay ``[B, T, H, K]``.
            b (torch.Tensor): Erase gate ``[B, T, H, K]``.
            w (torch.Tensor): Write gate ``[B, T, H, V]``.
            recurrent_state (torch.Tensor | None): Initial state.
            use_cache (bool): Whether to return the final state.
            cu_seqlens (Any): Cumulative sequence lengths.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: ``(output, final_state)``.
        """
        from qwendopamine.models.gdn2._prepare import dispatch_backend

        return dispatch_backend(
            self,
            backend,
            mode,
            q,
            k,
            v,
            g,
            b,
            w,
            recurrent_state,
            use_cache,
            cu_seqlens,
        )

    def _run_taichi_backend(
        self,
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
        r"""Run the Taichi kernel. Falls back to the torch path on failure."""
        if not _HAS_TAICHI_OPS:
            raise RuntimeError(
                "GDN-2 backend 'taichi' was requested but Taichi failed to "
                "initialise. Reinstall the project with `uv sync` to ensure "
                "the taichi dependency is present."
            )
        try:
            if mode == "chunk" and _taichi_chunk_gdn2 is not None:
                return _taichi_chunk_gdn2(
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
            if _taichi_recurrent_gdn2 is not None:
                return _taichi_recurrent_gdn2(
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
            raise RuntimeError("Taichi ops present but no kernel is callable")
        except (
            RuntimeError,
            TypeError,
            ValueError,
            AttributeError,
        ) as e:
            from qwendopamine.models.gdn2.backend import _warn_fallback_once

            _warn_fallback_once(
                f"Taichi kernel failed ({e}); falling back to pure PyTorch"
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
        r"""_compute_output(hidden_states: torch.Tensor, o: torch.Tensor, g: torch.Tensor, is_padded: bool, indices: torch.Tensor | None, batch: int, seq_len: int) -> torch.Tensor

        Delegate to :func:`qwendopamine.models.gdn2._output.compute_output`.

        Args:
            hidden_states (torch.Tensor): Input ``[B, T, D]``.
            o (torch.Tensor): Mixer output ``[B, T, H, V]``.
            g (torch.Tensor): Reserved (unused).
            is_padded (bool): Whether the input was padded.
            indices (torch.Tensor | None): Unpad indices when ``is_padded``.
            batch (int): Batch size.
            seq_len (int): Padded sequence length.

        Returns:
            torch.Tensor: Output ``[B, T, D]``.
        """
        from qwendopamine.models.gdn2._output import compute_output

        return compute_output(
            hidden_states,
            o,
            self.g_proj,
            self.o_norm,
            self.o_proj,
            self.head_v_dim,
            is_padded,
            indices,
            batch,
            seq_len,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | dict[str, Any] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | dict[str, Any] | None]:
        r"""Forward pass of the GDN-2 token-mixing layer.

        Args:
            hidden_states: Hidden-state ``[B, T, D]``.
            attention_mask: Padding mask ``[B, T]`` (optional).
            past_key_values: Cache for decoding.
            use_cache: Return updated cache.

        Returns:
            Tuple ``(hidden_states, attentions, past_key_values)``.
        """
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
            hidden_states,
            conv_state_q,
            conv_state_k,
            conv_state_v,
            use_cache if use_cache is not None else False,
            cu_seqlens,
        )
        q, k, v, g, b, w = self._prepare_tokens(q, k, v, g, b, w)

        backend = resolve_gdn2_backend(
            self.backend, training=self.training, seq_len=seq_len
        )
        o, recurrent_state = self._dispatch_backend(
            backend,
            mode,
            q,
            k,
            v,
            g,
            b,
            w,
            recurrent_state,
            use_cache if use_cache is not None else False,
            cu_seqlens,
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

        return (
            self._compute_output(
                hidden_states, o, g, is_padded, indices, batch, seq_len
            ),
            None,
            past_key_values,
        )


__all__ = ["GatedDeltaNet2"]
