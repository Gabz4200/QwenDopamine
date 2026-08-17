# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""GatedSurpriseNet: Precision-Weighted Surprise Fast-Weight Recurrence & Parallel Scan.

Mathematical Formulation:
--------------------------
1. Precision-Weighted Surprise Fast-Weight Update View:
   At each token step t, the memory S_t in R^{d_k x d_v} optimizes the local Surprise objective:
       L_t(S) = 0.5 * ||S - S_bar_t||_F^2 + 0.5 * < z_t - S^T (b_t * k_t), Pi_t (z_t - S^T (b_t * k_t)) >
   where:
       S_bar_t = Diag(alpha_t) S_{t-1}   (channel-wise decayed memory, alpha_t = exp(g_t))
       e_t = b_t * k_t                   (erase key with channel-wise erase gate b_t in [0, 1]^{d_k})
       z_t = w_t * v_t                   (write target with channel-wise write gate w_t in [0, 1]^{d_v})
       r_t = S_bar_t^T e_t               (memory read / predictive expectation)
       sigma_t^2 = softplus(W_sigma x_t) + eps  (token-level predicted variance/uncertainty metric)
       Pi_t = Diag(pi_t) = Diag(1 / sigma_t^2)  (Surprise Precision Metric)
       s_t = pi_t * (z_t - r_t)          (Surprise residual vector)

   Setting the matrix gradient nabla_S L_t(S) = 0 yields the exact closed-form update:
       S_t = S_bar_t + (b_t * k_t) s_t^T = S_bar_t + (b_t * k_t) (pi_t * (z_t - S_bar_t^T (b_t * k_t)))^T
   which in operator form on the memory matrix is:
       S_t = (I - (b_t * k_t) (pi_t * b_t * k_t)^T) S_bar_t + (b_t * k_t) (pi_t * w_t * v_t)^T
   and readout:
       o_t = S_t^T q_t.

2. Chunkwise Parallel WY Algorithm:
   Inside each chunk of length C:
       - Cumulative decay: G_r = sum_{i<=r} g_i, gamma_r = exp(G_r), gamma_C = exp(G_C)
       - Normalized keys: k_bar_r = gamma_r^{-1} * (b_r * k_r)
       - Normalized erase keys: e_bar_r = gamma_r * (pi_r * b_r * k_r)
       - Precision-scaled write target matrix: Z = pi * (w * v)
       - Strictly lower triangular matrix: T = tril(E_bar K_bar^T, -1) in R^{C x C}
       - Single C x C unit lower triangular WY solve per head:
             A = (I + T)^{-1} in R^{C x C}
             R = A (Z - E_bar S_0)
       - End-of-chunk state:
             S_C = Diag(gamma_C) S_0 + K_bar^T R
       - Chunk output:
             O = Q_gamma S_0 + A_qk R, where Q_gamma = gamma * Q, A_qk = tril(Q_gamma K_bar^T, 0).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from qwendopamine.models.gdn2.gdn2 import RMSNormGated, ShortConvolution

if TYPE_CHECKING:
    from transformers.cache_utils import Cache
else:
    try:
        from transformers.cache_utils import Cache
    except ImportError:
        Cache = Any

try:
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
except ImportError:
    LinearAttentionCacheLayerMixin = type(None)  # type: ignore[misc, assignment]

# Module-level single-warning guard for CPU fallback
_WARNED_FALLBACKS: set[str] = set()


def _warn_fallback_once(reason: str) -> None:
    if reason not in _WARNED_FALLBACKS:
        _WARNED_FALLBACKS.add(reason)
        warnings.warn(
            f"[surprise_net] Using pure PyTorch fallback: {reason}", stacklevel=2
        )


SURPRISE_NET_BACKENDS = (
    "auto",
    "torch",
    "torch-chunk",
    "torch-recurrent",
    "compiled",
    "triton",
    "fla",
)


def resolve_surprise_net_backend(
    requested: str,
    *,
    training: bool,
    seq_len: int,
) -> str:
    r"""Resolve the concrete execution backend for a GatedSurpriseNet forward call."""
    if requested not in SURPRISE_NET_BACKENDS:
        raise ValueError(
            f"Invalid GatedSurpriseNet backend '{requested}'. Valid backends: {list(SURPRISE_NET_BACKENDS)}"
        )
    if requested != "auto":
        return requested

    if not training and seq_len <= 1:
        return "torch-recurrent"
    if training:
        return "torch-chunk"
    if seq_len <= 64:
        return "torch-recurrent"
    return "torch-chunk"


@dataclass
class SurpriseRecurrenceState:
    r"""Recurrence state for GatedSurpriseNet containing fast-weight memory tensor."""

    memory: torch.Tensor
    first_moment: torch.Tensor | None = None
    second_moment: torch.Tensor | None = None
    step: torch.Tensor | None = None


def l2_normalize_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Normalizes a tensor along its final dimension using L2 norm vector scaling."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def gaussian_nll_diag(
    target: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
    full: bool = False,
) -> torch.Tensor:
    r"""Computes the diagonal Gaussian Negative Log-Likelihood diagnostic loss."""
    var = var.clamp_min(eps)
    loss = 0.5 * (torch.log(var) + (target - mean).square() / var)
    if full:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    return loss.sum(dim=-1)


def torch_get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Computes non-zero token indexing arrays and cumulative sequence lengths for unpadding."""
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = (
        int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() > 0 else 0
    )
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def torch_index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    r"""Indexes the outer axis of a multi-dimensional tensor using flat 1D indices."""
    return x[indices]


def torch_pad_input(
    hidden_states: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int
) -> torch.Tensor:
    r"""Pads a 1D unpadded sequence tensor back to 3D batch shape :math:`(B, L, D)`."""
    output_shape = (batch_size * seq_len,) + hidden_states.shape[1:]
    padded = torch.zeros(
        output_shape, dtype=hidden_states.dtype, device=hidden_states.device
    )
    padded[indices] = hidden_states
    return padded.view(batch_size, seq_len, *hidden_states.shape[1:])


class SurpriseMemory(nn.Module):
    r"""Algebraic Surprise fast-weight associative memory with serial & chunkwise WY scan."""

    def __init__(
        self,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        nll_var_eps: float = 1e-6,
        nll_full: bool = False,
        learnable_init: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.nll_var_eps = nll_var_eps
        self.nll_full = nll_full

        init = torch.zeros(num_heads, head_k_dim, head_v_dim)
        if learnable_init:
            self.memory_init = nn.Parameter(init)
        else:
            self.register_buffer("memory_init", init)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SurpriseRecurrenceState:
        device = device if device is not None else self.memory_init.device
        dtype = dtype if dtype is not None else torch.float32
        mem0 = (
            self.memory_init.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .clone()
        )
        return SurpriseRecurrenceState(memory=mem0)

    def one_step(
        self,
        state: SurpriseRecurrenceState,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        g_t: torch.Tensor,
        b_t: torch.Tensor,
        w_t: torch.Tensor,
        u_t: torch.Tensor | None = None,
        sigma_sq_t: torch.Tensor | None = None,
    ) -> tuple[SurpriseRecurrenceState, torch.Tensor, torch.Tensor]:
        r"""Compute one step of precision-weighted surprise fast-weight recurrence."""
        out_dtype = q_t.dtype
        q_f = q_t.float()
        k_f = k_t.float()
        v_f = v_t.float()
        g_f = g_t.float()
        b_f = b_t.float()
        w_f = w_t.float()

        if sigma_sq_t is not None:
            pi_t = 1.0 / (sigma_sq_t.float().clamp_min(1e-6))
        elif u_t is not None:
            pi_t = u_t.float()
        else:
            pi_t = torch.ones_like(w_f)

        alpha_t = torch.exp(g_f)
        S_bar = alpha_t.unsqueeze(-1) * state.memory.float()

        erase_key = b_f * k_f
        target_mu = w_f * v_f

        pred_mu = torch.einsum("bhkv,bhk->bhv", S_bar, erase_key)
        surprise_residual = pi_t * (target_mu - pred_mu)

        memory_new = S_bar + torch.einsum("bhk,bhv->bhkv", erase_key, surprise_residual)
        out_t = torch.einsum("bhkv,bhk->bhv", memory_new, q_f).to(out_dtype)

        pred_var = (
            sigma_sq_t.float()
            if sigma_sq_t is not None
            else (F.softplus(torch.log1p(pred_mu.square())) + self.nll_var_eps)
        )
        nll_t = gaussian_nll_diag(
            target=target_mu,
            mean=pred_mu,
            var=pred_var,
            eps=self.nll_var_eps,
            full=self.nll_full,
        ).to(out_dtype)

        new_state = SurpriseRecurrenceState(memory=memory_new)
        return new_state, out_t, nll_t

    def serial_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor | None = None,
        sigma_sq: torch.Tensor | None = None,
        initial_state: SurpriseRecurrenceState | None = None,
        detach_state_every_step: bool = False,
    ) -> tuple[torch.Tensor, SurpriseRecurrenceState, torch.Tensor]:
        r"""Token-by-token recurrence serial scan."""
        bs = k.shape[0]
        if initial_state is None:
            state = self.initial_state(bs, device=k.device, dtype=torch.float32)
        else:
            state = SurpriseRecurrenceState(memory=initial_state.memory.float())

        outputs: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        seq_len = k.shape[1]

        for i in range(seq_len):
            u_i = u[:, i] if u is not None else None
            sig_i = sigma_sq[:, i] if sigma_sq is not None else None
            state, out_i, nll_i = self.one_step(
                state, q[:, i], k[:, i], v[:, i], g[:, i], b[:, i], w[:, i], u_i, sig_i
            )
            outputs.append(out_i)
            losses.append(nll_i)
            if detach_state_every_step:
                state = SurpriseRecurrenceState(memory=state.memory.detach())

        return torch.stack(outputs, dim=1), state, torch.stack(losses, dim=1)

    def chunk_parallel_training_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor | None = None,
        sigma_sq: torch.Tensor | None = None,
        chunk_size: int = 128,
        initial_state: SurpriseRecurrenceState | None = None,
    ) -> tuple[torch.Tensor, SurpriseRecurrenceState, torch.Tensor]:
        r"""Chunkwise WY algorithm for Precision-Weighted Surprise recurrence."""
        bs, ts = k.shape[:2]
        out_dtype = q.dtype

        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        g_f = g.float()
        b_f = b.float()
        w_f = w.float()

        if sigma_sq is not None:
            sigma_sq_f = sigma_sq.float()
            pi_f = 1.0 / (sigma_sq_f.clamp_min(1e-6))
        elif u is not None:
            pi_f = u.float()
            sigma_sq_f = None
        else:
            pi_f = torch.ones_like(w_f)
            sigma_sq_f = None

        state = (
            self.initial_state(bs, device=k.device, dtype=torch.float32)
            if initial_state is None
            else SurpriseRecurrenceState(memory=initial_state.memory.float())
        )
        S_chunk = state.memory

        outputs: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []

        if not hasattr(self, "_eye_cache") or not isinstance(self._eye_cache, dict):
            self._eye_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

        for start in range(0, ts, chunk_size):
            end = min(start + chunk_size, ts)
            c_len = end - start

            qc = q_f[:, start:end]
            kc = k_f[:, start:end]
            vc = v_f[:, start:end]
            gc = g_f[:, start:end]
            bc = b_f[:, start:end]
            wc = w_f[:, start:end]
            pic = pi_f[:, start:end]
            sigc = sigma_sq_f[:, start:end] if sigma_sq_f is not None else None

            S_0 = S_chunk

            G = torch.cumsum(gc, dim=1)
            gamma = torch.exp(G)
            gamma_last = gamma[:, -1]

            ekc = bc * kc
            kbar = ekc / gamma.clamp_min(1e-12)
            ebar = gamma * ekc
            Zc = wc * vc

            E_mat = ebar.permute(0, 2, 1, 3)
            K_mat = kbar.permute(0, 2, 1, 3)

            T_mat = torch.tril(
                torch.matmul(E_mat, K_mat.transpose(-1, -2)), diagonal=-1
            )

            S_0_v = S_0.permute(0, 1, 3, 2)
            ES0 = torch.matmul(S_0_v, E_mat.transpose(-1, -2))
            Z_mat = Zc.permute(0, 2, 3, 1)

            R_raw = Z_mat - ES0
            Pi_mat = pic.permute(0, 2, 3, 1)
            RHS = (Pi_mat * R_raw).unsqueeze(-1)

            T_scaled = Pi_mat.unsqueeze(-1) * T_mat.unsqueeze(2)

            cache_key = (c_len, q.device)
            if cache_key not in self._eye_cache:
                self._eye_cache[cache_key] = torch.eye(
                    c_len, device=q.device, dtype=torch.float32
                )
            eye = self._eye_cache[cache_key].view(1, 1, 1, c_len, c_len)
            L = eye + T_scaled

            # Flatten 5D (B, H, d_v, C, C) -> 3D (B*H*d_v, C, C) for cuBLAS 3D batched solve
            num_systems = bs * self.num_heads * self.head_v_dim
            L_3d = L.reshape(num_systems, c_len, c_len)
            RHS_3d = RHS.reshape(num_systems, c_len, 1)

            R_vec_3d = torch.linalg.solve_triangular(
                L_3d, RHS_3d, upper=False
            )
            R_vec = R_vec_3d.view(bs, self.num_heads, self.head_v_dim, c_len)
            R = R_vec.permute(0, 1, 3, 2)

            gamma_last_exp = gamma_last.unsqueeze(1)
            K_tail = ((gamma_last_exp / gamma.clamp_min(1e-12)) * ekc).permute(
                0, 2, 1, 3
            )
            S_chunk = gamma_last.unsqueeze(-1) * S_0 + torch.matmul(
                K_tail.transpose(-1, -2), R
            )

            Q_gamma = (gamma * qc).permute(0, 2, 1, 3)
            out_intra_prev = torch.matmul(Q_gamma, S_0)
            A_qk = torch.tril(
                torch.matmul(Q_gamma, K_mat.transpose(-1, -2)), diagonal=0
            )
            out_intra_new = torch.matmul(A_qk, R)

            O_chunk = (out_intra_prev + out_intra_new).permute(0, 2, 1, 3)
            outputs.append(O_chunk.to(out_dtype))

            TR_vec = torch.matmul(R.permute(0, 1, 3, 2), T_mat.transpose(-1, -2))
            pred_mu_c = (ES0 + TR_vec).permute(0, 3, 1, 2)
            target_mu_c = wc * vc
            pred_var_c = (
                sigc
                if sigc is not None
                else (F.softplus(torch.log1p(pred_mu_c.square())) + self.nll_var_eps)
            )
            nll_c = gaussian_nll_diag(
                target=target_mu_c,
                mean=pred_mu_c,
                var=pred_var_c,
                eps=self.nll_var_eps,
                full=self.nll_full,
            ).to(out_dtype)
            losses.append(nll_c)

        final_state = SurpriseRecurrenceState(memory=S_chunk)
        return torch.cat(outputs, dim=1), final_state, torch.cat(losses, dim=1)


# Aliases for backward compatibility
SurpriseMemoryAdam = SurpriseMemory


class GatedSurpriseNet(nn.Module):
    r"""GatedSurpriseNet token mixer using precision-weighted surprise fast-weight recurrence.

    This module provides a drop-in token mixer matching QwenDopamine transformer blocks
    implementing local Surprise optimization and precision-weighted fast-weight memory updates.
    """

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
        nll_var_eps: float = 1e-6,
        nll_full: bool = False,
        learnable_init_state: bool = False,
        train_chunk_size: int = 128,
        backend: str = "auto",
        compile_backend: bool = False,
        max_write_bound: float = 1.50,
        max_erase_bound: float = 3.00,
        max_precision_bound: float = 2.00,
        **kwargs: Any,
    ) -> None:
        super().__init__()

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
            nll_var_eps = getattr(cfg, "nll_var_eps", nll_var_eps)
            nll_full = getattr(cfg, "nll_full", nll_full)
            learnable_init_state = getattr(
                cfg, "learnable_init_state", learnable_init_state
            )
            train_chunk_size = getattr(
                cfg, "train_chunk_size", getattr(cfg, "chunk_size", train_chunk_size)
            )
            backend = getattr(cfg, "backend", backend)
            compile_backend = getattr(cfg, "compile_backend", compile_backend)
            max_write_bound = getattr(cfg, "max_write_bound", max_write_bound)
            max_erase_bound = getattr(cfg, "max_erase_bound", max_erase_bound)
            max_precision_bound = getattr(cfg, "max_precision_bound", max_precision_bound)
        elif isinstance(hidden_size_or_config, dict):
            cfg_dict = hidden_size_or_config
            hidden_size = cfg_dict.get("hidden_size", 2048)
            num_heads = cfg_dict.get("num_heads", 16)
            head_dim = cfg_dict.get("head_dim", 128)
            num_v_heads = cfg_dict.get("num_v_heads", num_heads)
            conv_size = cfg_dict.get("conv_size", conv_size)
            norm_eps = cfg_dict.get("norm_eps", norm_eps)
            allow_neg_eigval = cfg_dict.get("allow_neg_eigval", allow_neg_eigval)
            expand_v = cfg_dict.get("expand_v", expand_v)
            nll_var_eps = cfg_dict.get("nll_var_eps", nll_var_eps)
            nll_full = cfg_dict.get("nll_full", nll_full)
            learnable_init_state = cfg_dict.get(
                "learnable_init_state", learnable_init_state
            )
            train_chunk_size = cfg_dict.get("train_chunk_size", train_chunk_size)
            backend = cfg_dict.get("backend", backend)
            compile_backend = cfg_dict.get("compile_backend", compile_backend)
            max_write_bound = cfg_dict.get("max_write_bound", max_write_bound)
            max_erase_bound = cfg_dict.get("max_erase_bound", max_erase_bound)
            max_precision_bound = cfg_dict.get("max_precision_bound", max_precision_bound)
        elif hidden_size is None:
            hidden_size = int(hidden_size_or_config)

        self.hidden_size = hidden_size
        self.num_heads = num_heads if num_heads is not None else 16
        self.head_dim = head_dim if head_dim is not None else 128
        self.num_v_heads = num_v_heads if num_v_heads is not None else self.num_heads
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.layer_idx = layer_idx
        self.norm_eps = norm_eps

        self.nll_var_eps = nll_var_eps
        self.nll_full = nll_full
        self.learnable_init_state = learnable_init_state
        self.train_chunk_size = train_chunk_size
        self.backend = backend
        self.compile_backend = compile_backend
        self.max_write_bound = float(max_write_bound)
        self.max_erase_bound = float(max_erase_bound)
        self.max_precision_bound = float(max_precision_bound)

        self.mode = mode
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)

        if not math.isclose(
            self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5
        ):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )
        if not math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={self.head_dim}."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # Variance / Uncertainty projection for precision weighting
        self.var_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )

        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0.1, 2.0))
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

        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNormGated(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.memory = SurpriseMemory(
            num_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            nll_var_eps=self.nll_var_eps,
            nll_full=self.nll_full,
            learnable_init=self.learnable_init_state,
        )

        self.apply(self._initialize_weights)

        # Initialize var_proj final linear bias to 0.0 after apply() so initial
        # precision pi ~ 1.0 (max_precision_bound * sigmoid(0) = 2.0 * 0.5 = 1.0)
        # without running redundant zeroing on every submodule in the tree.
        if isinstance(self.var_proj[-1], nn.Linear) and self.var_proj[-1].bias is not None:
            nn.init.zeros_(self.var_proj[-1].bias)

    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNormGated):
            nn.init.ones_(module.weight)
        cast(Any, module)._is_hf_initialized = True

    def _get_cache(
        self, past_key_values: Cache | dict[str, Any] | None
    ) -> tuple[
        SurpriseRecurrenceState | torch.Tensor | None,
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
            return past_key_values.get("recurrent_state"), past_key_values.get(
                "conv_state"
            )

        return None, None

    def _update_cache(
        self,
        past_key_values: Cache | dict[str, Any] | None,
        recurrent_state: SurpriseRecurrenceState | torch.Tensor | None,
        conv_state: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
        | None,
    ) -> None:
        if past_key_values is None:
            return

        state_tensor = (
            recurrent_state.memory
            if isinstance(recurrent_state, SurpriseRecurrenceState)
            else recurrent_state
        )

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
                    and state_tensor is not None
                ):
                    try:
                        past_key_values.update_recurrent_state(
                            state_tensor, self.layer_idx
                        )
                    except (TypeError, ValueError, AttributeError, RuntimeError, IndexError) as e:
                        _warn_fallback_once(f"update_recurrent_state failed: {e}")
                elif state_tensor is not None:
                    rec_dict = getattr(layer_cache, "recurrent_states", None)
                    if isinstance(rec_dict, dict):
                        rec_dict[0] = state_tensor
                    elif hasattr(layer_cache, "recurrent_state"):
                        layer_cache.recurrent_state = state_tensor

                if (
                    is_recurrent_layer
                    and hasattr(past_key_values, "update_conv_state")
                    and conv_state is not None
                ):
                    try:
                        past_key_values.update_conv_state(
                            cast(Any, conv_state), self.layer_idx
                        )
                    except (TypeError, ValueError, AttributeError, RuntimeError, IndexError) as e:
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

    def _project_inputs(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        use_cache: bool = False,
        conv_states: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ]
        | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
    ]:
        new_conv_q, new_conv_k, new_conv_v = None, None, None
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = (
                conv_states if conv_states is not None else (None, None, None)
            )
            q, new_conv_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, new_conv_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, new_conv_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = (
            -self.A_log.float().exp().repeat_interleave(self.head_k_dim)
            * F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        ).to(hidden_states.dtype)
        b = self.max_erase_bound * self.b_proj(hidden_states).sigmoid()
        w = self.max_write_bound * self.w_proj(hidden_states).sigmoid()
        pi = self.max_precision_bound * self.var_proj(hidden_states).sigmoid()
        sigma_sq = 1.0 / pi.clamp_min(1e-6)

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b_gate = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w_gate = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)
        sigma_sq = rearrange(sigma_sq, "... (h d) -> ... h d", d=self.head_v_dim)

        q = l2_normalize_last(q)
        k = l2_normalize_last(k)

        if self.num_v_heads > self.num_heads:
            groups = self.num_v_heads // self.num_heads
            q, k, g, b_gate = (
                repeat(x, "... h d -> ... (h g) d", g=groups) for x in (q, k, g, b_gate)
            )

        if self.allow_neg_eigval:
            b_gate = b_gate * 2.0

        new_conv_states = (
            (new_conv_q, new_conv_k, new_conv_v) if self.use_short_conv else None
        )
        return q, k, v, g, b_gate, w_gate, sigma_sq, new_conv_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | dict[str, Any] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | dict[str, Any] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding)."
            )

        batch_size, q_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None

        if attention_mask is not None:
            indices, cu_seqlens, _ = torch_get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = torch_index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        raw_rec_state, conv_states = self._get_cache(past_key_values)
        if isinstance(raw_rec_state, SurpriseRecurrenceState):
            recurrent_state = raw_rec_state
        elif isinstance(raw_rec_state, torch.Tensor):
            recurrent_state = SurpriseRecurrenceState(memory=raw_rec_state)
        else:
            recurrent_state = None

        should_use_cache = bool(use_cache or past_key_values is not None)

        q, k, v, g, b_gate, w_gate, sigma_sq, new_conv_states = self._project_inputs(
            hidden_states,
            cu_seqlens=cu_seqlens,
            use_cache=should_use_cache,
            conv_states=conv_states,
        )

        mode = resolve_surprise_net_backend(
            self.backend, training=self.training, seq_len=q_len
        )

        out: torch.Tensor | None = None
        final_recurrent_state: SurpriseRecurrenceState | None = None

        if mode in ("triton", "fla"):
            try:
                from qwendopamine.models.gated_surprise_net_ops import (
                    chunk_gated_surprise_net as _chunk_gated_surprise_net_op,
                )

                pi_tensor = 1.0 / (sigma_sq.float().clamp_min(1e-6))
                init_mem = recurrent_state.memory if recurrent_state is not None else None
                out_op, final_mem = _chunk_gated_surprise_net_op(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    b=b_gate,
                    w=w_gate,
                    pi=pi_tensor,
                    chunk_size=self.train_chunk_size,
                    initial_state=init_mem,
                )
                out = out_op
                final_recurrent_state = SurpriseRecurrenceState(memory=final_mem)
            except (RuntimeError, TypeError, ValueError, ImportError) as exc:
                _warn_fallback_once(
                    f"GatedSurpriseNet custom ops execution warning ({exc}); falling back to PyTorch chunk scan"
                )
                mode = "torch-chunk" if self.training else "torch-recurrent"

        if mode not in ("triton", "fla"):
            if mode in ("torch-chunk", "compiled", "auto"):
                out, final_recurrent_state, _ = (
                    self.memory.chunk_parallel_training_scan(
                        q=q,
                        k=k,
                        v=v,
                        g=g,
                        b=b_gate,
                        w=w_gate,
                        sigma_sq=sigma_sq,
                        chunk_size=self.train_chunk_size,
                        initial_state=recurrent_state,
                    )
                )
            else:
                out, final_recurrent_state, _ = self.memory.serial_scan(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    b=b_gate,
                    w=w_gate,
                    sigma_sq=sigma_sq,
                    initial_state=recurrent_state,
                    detach_state_every_step=False,
                )

        if should_use_cache:
            self._update_cache(
                past_key_values,
                recurrent_state=final_recurrent_state,
                conv_state=new_conv_states,
            )

        gate = rearrange(
            self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
        )
        assert out is not None
        out = self.o_norm(out, gate)
        out = rearrange(out, "... h d -> ... (h d)")
        out = self.o_proj(out)

        if attention_mask is not None and indices is not None:
            out = torch_pad_input(out.squeeze(0), indices, batch_size, q_len)

        return out, None, past_key_values


# Aliases for backward compatibility
GatedSurpriseNetAdam = GatedSurpriseNet
GatedSurpriseNetBlock = GatedSurpriseNet

__all__ = [
    "GatedSurpriseNet",
    "GatedSurpriseNetAdam",
    "GatedSurpriseNetBlock",
    "SurpriseMemory",
    "SurpriseMemoryAdam",
    "SurpriseRecurrenceState",
    "gaussian_nll_diag",
    "l2_normalize_last",
    "torch_get_unpad_data",
    "torch_index_first_axis",
    "torch_pad_input",
]
