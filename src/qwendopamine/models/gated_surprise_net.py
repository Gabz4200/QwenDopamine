"""GatedSurpriseNet: Closed-Form Algebraic Fast-Weight Recurrence on Surprise Optimization.

Mathematical Formulation:
--------------------------
1. Fast-Weight Update View:
   At each token step t, the memory S_t in R^{d_k x d_v} optimizes the local Surprise objective:
       L_t(S) = 0.5 * ||S - S_bar_t||_F^2 - <S k_t, s_t>
   where:
       S_bar_t = Diag(alpha_t) S_{t-1}   (channel-wise decayed memory, alpha_t = exp(g_t))
       e_t = b_t * k_t                   (erase key with channel-wise erase gate b_t in [0, 1]^{d_k})
       z_t = w_t * v_t                   (write target with channel-wise write gate w_t in [0, 1]^{d_v})
       r_t = S_bar_t^T e_t               (memory read / predictive expectation)
       u_t = sigma(W_u x_t)              (data-dependent surprise / precision gate in [0, 1]^{d_v})
       s_t = u_t * (z_t - r_t)           (Surprise vector / precision-weighted prediction residual)

   Setting the matrix gradient nabla_S L_t(S) = 0 yields the exact closed-form algebraic update:
       S_t = S_bar_t + k_t s_t^T = S_bar_t + k_t (u_t * (z_t - S_bar_t^T e_t))^T
   which in operator form on each value channel j in {1, ..., d_v} is:
       S_{t, :, j} = (I - u_{t, j} k_t e_t^T) S_bar_{t, :, j} + k_t (u_{t, j} z_{t, j})
   and readout:
       o_t = S_t^T q_t.

2. Chunkwise WY Algorithm with Channel-wise Decay & Surprise Inverse:
   Inside each chunk of length C:
       - Cumulative decay: G_r = sum_{i<=r} g_i, gamma_r = exp(G_r), gamma_C = exp(G_C)
       - Normalized keys: k_bar_r = gamma_r^{-1} * k_r, e_bar_r = gamma_r * (b_r * k_r)
       - Target matrix: Z = W * V
       - Strictly lower triangular matrix: T = tril(E_bar K_bar^T, -1) in R^{C x C}
       - Channel-wise Surprise WY inverse system:
         For each value channel j:
             L^{(j)} = I_C + Diag(u_{:, j}) T   (unit lower triangular)
             RHS_{:, j} = Diag(u_{:, j}) (Z_{:, j} - E_bar S_{0, :, j})
             R_{:, j} = (L^{(j)})^{-1} RHS_{:, j}
       - End-of-chunk state:
             S_C = Diag(gamma_C) S_0 + K_tail^T R, where (K_tail)_{r, :} = (gamma_C / gamma_r) * k_r
       - Chunk output:
             O = Q_gamma S_0 + A_qk R, where Q_gamma = gamma * Q, A_qk = tril(Q_gamma K_bar^T, 0).
"""

from __future__ import annotations

import math
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


@dataclass
class SurpriseRecurrenceState:
    r"""Recurrence state for GatedSurpriseNet containing fast-weight memory tensor."""

    memory: torch.Tensor
    first_moment: torch.Tensor | None = None
    second_moment: torch.Tensor | None = None
    step: torch.Tensor | None = None


def l2_normalize_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""l2_normalize_last(x, eps=1e-6) -> Tensor

    Normalizes a tensor along its final dimension using L2 norm vector scaling.

    Args:
        x (Tensor): Input tensor of arbitrary shape :math:`(..., D)`.
        eps (float, optional): Epsilon value to clamp minimum norm for numerical stability. Default: ``1e-6``.

    Returns:
        Tensor: L2-normalized tensor of same shape as ``x``.
    """
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def gaussian_nll_diag(
    target: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
    full: bool = False,
) -> torch.Tensor:
    r"""gaussian_nll_diag(target, mean, var, eps=1e-6, full=False) -> Tensor

    Computes the diagonal Gaussian Negative Log-Likelihood diagnostic loss.

    .. math::
        \mathcal{L} = \frac{1}{2} \sum_{i} \left( \log(\sigma_i^2) + \frac{(y_i - \mu_i)^2}{\sigma_i^2} \right)

    Args:
        target (Tensor): Ground truth target tensor.
        mean (Tensor): Predicted mean vector tensor.
        var (Tensor): Predicted variance vector tensor.
        eps (float, optional): Epsilon threshold for variance clamping. Default: ``1e-6``.
        full (bool, optional): If ``True``, includes the constant factor :math:`\frac{1}{2}\log(2\pi)`.
            Default: ``False``.

    Returns:
        Tensor: Summed negative log-likelihood loss along the feature dimension.
    """
    var = var.clamp_min(eps)
    loss = 0.5 * (torch.log(var) + (target - mean).square() / var)
    if full:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    return loss.sum(dim=-1)


def torch_get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""torch_get_unpad_data(attention_mask) -> (Tensor, Tensor, int)

    Computes non-zero token indexing arrays and cumulative sequence lengths for unpadding 2D sequence masks.

    Args:
        attention_mask (Tensor): Binary mask tensor of shape :math:`(B, L)`.

    Returns:
        tuple[Tensor, Tensor, int]: Tuple containing non-zero indices, cumulative sequence length tensor,
            and maximum sequence length integer in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = (
        int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() > 0 else 0
    )
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def torch_index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    r"""torch_index_first_axis(x, indices) -> Tensor

    Indexes the outer axis of a multi-dimensional tensor using flat 1D indices.

    Args:
        x (Tensor): Input tensor.
        indices (Tensor): Flat index tensor.

    Returns:
        Tensor: Gathered tensor sliced along axis 0.
    """
    return x[indices]


def torch_pad_input(
    hidden_states: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int
) -> torch.Tensor:
    r"""torch_pad_input(hidden_states, indices, batch_size, seq_len) -> Tensor

    Pads a 1D unpadded sequence tensor back to 3D batch shape :math:`(B, L, D)`.

    Args:
        hidden_states (Tensor): Flattened non-padded sequence tensor.
        indices (Tensor): Original token index tensor.
        batch_size (int): Target batch size :math:`B`.
        seq_len (int): Target sequence length :math:`L`.

    Returns:
        Tensor: Padded 3D sequence tensor.
    """
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
    ) -> tuple[SurpriseRecurrenceState, torch.Tensor, torch.Tensor]:
        r"""Compute one step of closed-form algebraic surprise fast-weight recurrence."""
        out_dtype = q_t.dtype
        q_f = q_t.float()
        k_f = k_t.float()
        v_f = v_t.float()
        g_f = g_t.float()
        b_f = b_t.float()
        w_f = w_t.float()
        u_f = u_t.float() if u_t is not None else torch.ones_like(w_f)

        alpha_t = torch.exp(g_f)
        S_bar = alpha_t.unsqueeze(-1) * state.memory.float()

        erase_key = b_f * k_f
        target_mu = w_f * v_f

        pred_mu = torch.einsum("bhkv,bhk->bhv", S_bar, erase_key)
        surprise_residual = u_f * (target_mu - pred_mu)

        memory_new = S_bar + torch.einsum("bhk,bhv->bhkv", k_f, surprise_residual)
        out_t = torch.einsum("bhkv,bhk->bhv", memory_new, q_f).to(out_dtype)

        pred_var = F.softplus(torch.log1p(pred_mu.square())) + self.nll_var_eps
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
            state, out_i, nll_i = self.one_step(
                state, q[:, i], k[:, i], v[:, i], g[:, i], b[:, i], w[:, i], u_i
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
        chunk_size: int = 128,
        initial_state: SurpriseRecurrenceState | None = None,
    ) -> tuple[torch.Tensor, SurpriseRecurrenceState, torch.Tensor]:
        r"""Chunkwise WY algorithm for Surprise recurrence with channel-wise decay and WY inverse."""
        bs, ts = k.shape[:2]
        out_dtype = q.dtype

        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        g_f = g.float()
        b_f = b.float()
        w_f = w.float()
        u_f = u.float() if u is not None else torch.ones_like(w_f)

        state = (
            self.initial_state(bs, device=k.device, dtype=torch.float32)
            if initial_state is None
            else SurpriseRecurrenceState(memory=initial_state.memory.float())
        )
        S_chunk = state.memory

        outputs: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []

        for start in range(0, ts, chunk_size):
            end = min(start + chunk_size, ts)
            c_len = end - start

            qc = q_f[:, start:end]
            kc = k_f[:, start:end]
            vc = v_f[:, start:end]
            gc = g_f[:, start:end]
            bc = b_f[:, start:end]
            wc = w_f[:, start:end]
            uc = u_f[:, start:end]

            S_0 = S_chunk

            G = torch.cumsum(gc, dim=1)
            gamma = torch.exp(G)
            gamma_last = gamma[:, -1]

            k_bar = kc / gamma.clamp_min(1e-12)
            e_bar = gamma * (bc * kc)
            Zc = wc * vc

            E_mat = e_bar.permute(0, 2, 1, 3)
            K_mat = k_bar.permute(0, 2, 1, 3)

            T_mat = torch.tril(
                torch.matmul(E_mat, K_mat.transpose(-1, -2)), diagonal=-1
            )

            u_mat = uc.permute(0, 2, 3, 1)
            u_T = u_mat.unsqueeze(-1) * T_mat.unsqueeze(2)
            eye = torch.eye(c_len, device=q.device, dtype=torch.float32).view(
                1, 1, 1, c_len, c_len
            )
            L = eye + u_T

            S_0_v = S_0.permute(0, 1, 3, 2)
            ES0 = torch.matmul(S_0_v, E_mat.transpose(-1, -2))
            Z_mat = Zc.permute(0, 2, 3, 1)

            RHS = u_mat * (Z_mat - ES0)
            R_vec = torch.linalg.solve_triangular(
                L, RHS.unsqueeze(-1), upper=False
            ).squeeze(-1)
            R = R_vec.permute(0, 1, 3, 2)

            gamma_last_exp = gamma_last.unsqueeze(1)
            K_tail = ((gamma_last_exp / gamma.clamp_min(1e-12)) * kc).permute(
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

            TR = torch.matmul(T_mat, R)
            pred_mu_c = (ES0.permute(0, 1, 3, 2) + TR).permute(0, 2, 1, 3)
            pred_var_c = F.softplus(torch.log1p(pred_mu_c.square())) + self.nll_var_eps
            nll_c = gaussian_nll_diag(
                target=Zc,
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
    r"""GatedSurpriseNet token mixer using closed-form algebraic surprise fast-weight recurrence.

    This module provides a drop-in token mixer matching QwenDopamine transformer blocks
    while implementing the algebraic closed-form Surprise gradient descent and chunkwise WY algorithm.
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
            train_chunk_size = getattr(cfg, "train_chunk_size", train_chunk_size)
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
        self.u_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

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

    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        cast(Any, module)._is_hf_initialized = True

    def _get_cache(
        self, past_key_values: Cache | dict[str, Any] | None
    ) -> tuple[
        SurpriseRecurrenceState | None,
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
    ]:
        if past_key_values is None:
            return None, None

        if isinstance(past_key_values, Cache):
            layers = getattr(past_key_values, "layers", [])
            if self.layer_idx is not None and self.layer_idx < len(layers):
                layer_cache = layers[self.layer_idx]
                rec_states = getattr(layer_cache, "recurrent_states", None)
                rec_state = (
                    rec_states[0]
                    if rec_states is not None and len(rec_states) > 0
                    else None
                )
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
        recurrent_state: SurpriseRecurrenceState | None,
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
                    past_key_values.update_recurrent_state(
                        cast(Any, recurrent_state), self.layer_idx
                    )
                if (
                    is_recurrent_layer
                    and hasattr(past_key_values, "update_conv_state")
                    and conv_state is not None
                ):
                    past_key_values.update_conv_state(
                        cast(Any, conv_state), self.layer_idx
                    )
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
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()
        u = self.u_proj(hidden_states).sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b_gate = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w_gate = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)
        u_gate = rearrange(u, "... (h d) -> ... h d", d=self.head_v_dim)

        q = l2_normalize_last(q)
        k = l2_normalize_last(k)

        if self.num_v_heads > self.num_heads:
            groups = self.num_v_heads // self.num_heads
            q, k, g, b_gate = (
                repeat(x, "... h d -> ... (h g) d", g=groups) for x in (q, k, g, b_gate)
            )

        if self.allow_neg_eigval:
            b_gate = b_gate * 2.0
            u_gate = u_gate * 2.0

        new_conv_states = (
            (new_conv_q, new_conv_k, new_conv_v) if self.use_short_conv else None
        )
        return q, k, v, g, b_gate, w_gate, u_gate, new_conv_states

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
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None

        if attention_mask is not None:
            indices, cu_seqlens, _ = torch_get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = torch_index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        recurrent_state, conv_states = self._get_cache(past_key_values)
        should_use_cache = bool(use_cache or past_key_values is not None)

        q, k, v, g, b_gate, w_gate, u_gate, new_conv_states = self._project_inputs(
            hidden_states,
            cu_seqlens=cu_seqlens,
            use_cache=should_use_cache,
            conv_states=conv_states,
        )

        if self.training:
            mode = "chunk"
        else:
            mode = (
                "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
            )

        if mode == "chunk":
            out, final_recurrent_state, _ = self.memory.chunk_parallel_training_scan(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b_gate,
                w=w_gate,
                u=u_gate,
                chunk_size=self.train_chunk_size,
                initial_state=recurrent_state,
            )
        elif mode == "fused_recurrent":
            out, final_recurrent_state, _ = self.memory.serial_scan(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b_gate,
                w=w_gate,
                u=u_gate,
                initial_state=recurrent_state,
                detach_state_every_step=False,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if should_use_cache:
            self._update_cache(
                past_key_values,
                recurrent_state=final_recurrent_state,
                conv_state=new_conv_states,
            )

        gate = rearrange(
            self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
        )
        out = self.o_norm(out, gate)
        out = rearrange(out, "b t h d -> b t (h d)")
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
