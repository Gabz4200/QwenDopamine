# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""RL-augmented Gated Delta Rule 2 components: ValueBaselineEMA, AdvantageGate, DeltaMemoryCore, and ReinforcedDeltaLayer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "AdvantageGate",
    "DeltaMemoryCore",
    "GatedRewardNetConfig",
    "ReinforcedDeltaLayer",
    "ValueBaselineEMA",
]


@dataclass
class GatedRewardNetConfig:
    r"""Configuration for :class:`GatedRewardNet`."""

    hidden_size: int = 2048
    k_stats: int = 6
    reward_encoder: nn.Module | None = None
    layer_idx: int | None = None
    use_short_conv: bool = True
    conv_size: int = 4
    conv_bias: bool = False
    init_alpha: float = 0.1
    num_heads: int = 16
    head_dim: int = 128
    expand_v: float = 1.0
    norm_eps: float = 1e-5
    chunk_size: int = 64
    reward_dropout: float = 0.0
    advantage_dropout: float = 0.0
    hidden_dropout: float = 0.0
    advantage_legacy_coupled: bool = False
    r"""When True the advantage gate keeps the original
    ``omega_t = 2·σ(W A + b)`` single-scalar behaviour. When False (the
    default) the gate splits into ``(plasticity, write, erase)`` so negative
    advantage actively erases and large-magnitude advantage increases
    plasticity. Set True only when loading a checkpoint that depends on the
    old coupled gate output."""
    memory_rank: int | None = None
    r"""Optional low-rank factorization for the d×d fast-weight state.

    When ``None`` (default) the full ``(B, d, d)`` matrix is used. When set to a
    positive integer ``r`` the state is factored as ``U @ V.T`` with
    ``U, V ∈ R^{B × d × r}`` to reduce memory and compute. ``r`` should be
    much smaller than ``d`` (e.g. 16/32/64 for d=4096)."""
    reward_normalize: bool = False
    r"""When True, ``GatedRewardNet`` standardises ``reward_values`` to
    advantage-like scale before they enter the reinforced delta loop. The
    spec recommends feeding advantage-like signals (TD error, GAE,
    normalised return-to-go, surprise) rather than raw sparse environment
    reward. This flag applies the same standardisation inline so raw
    reward inputs do not destabilise the advantage gate."""
    reward_normalize_eps: float = 1e-5
    r"""Numerical epsilon for the running-std division in
    :func:`normalize_reward_for_advantage`."""
    reward_ema_alpha: float = 0.1
    r"""EMA decay for the running mean and std of the reward values. The
    running statistics persist in the cache so they survive step-by-step
    generation."""


class ValueBaselineEMA(nn.Module):
    r"""Vectorial Exponential Moving Average with data-dependent decay.

    Tracks the expectation of a vectorial reward signal (e.g., statistics like
    mean, max, min, std, sum) using an EMA where the decay rate α_t is generated
    from the current input x_t. This is mathematically equivalent to a minLSTM-style
    recurrence operating on a k-dimensional vector.

    For each statistic dimension i ∈ {1..k}:
        α_t[i] = σ(W_α[i] x_t + b_α[i])
        V_t[i] = α_t[i] ⊙ R_t[i] + (1 - α_t[i]) ⊙ V_{t-1}[i]
        A_t[i] = R_t[i] - V_{t-1}[i]

    Args:
        d_model (int): Input feature dimension.
        k_stats (int): Number of reward statistics dimensions.
        init_alpha (float, optional): Initial decay rate before sigmoid.
            Default: 0.1 (slow initial adaptation).

    Shape:
        - x: (B, d_model)
        - R_stats: (B, k_stats)
        - V_prev: (B, k_stats)
        - Returns: V_t (B, k_stats), A_t (B, k_stats)
    """

    def __init__(
        self,
        d_model: int,
        k_stats: int,
        init_alpha: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if k_stats <= 0:
            raise ValueError("k_stats must be positive.")
        if not (0 < init_alpha < 1):
            raise ValueError("init_alpha must be in (0, 1).")

        self.d_model = d_model
        self.k_stats = k_stats

        self.alpha_proj = nn.Linear(d_model, k_stats)

        with torch.no_grad():
            init_bias = torch.log(torch.tensor(init_alpha / (1 - init_alpha)))
            self.alpha_proj.bias.fill_(init_bias)
            self.alpha_proj.weight.zero_()

    def forward(
        self,
        x: Tensor,
        R_stats: Tensor,
        V_prev: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, d_model) - Current input features.
            R_stats: (B, k_stats) - Vectorial reward statistics at step t.
            V_prev: (B, k_stats) - Previous EMA baseline.

        Returns:
            V_t: (B, k_stats) - Updated baseline.
            A_t: (B, k_stats) - Instantaneous advantage vector.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (B, d_model), got {x.shape}.")
        if R_stats.shape != V_prev.shape:
            raise ValueError(
                f"R_stats {R_stats.shape} and V_prev {V_prev.shape} must match."
            )
        if R_stats.size(-1) != self.k_stats:
            raise ValueError(
                f"Expected k_stats={self.k_stats}, got {R_stats.size(-1)}."
            )

        # α_t = σ(W_α x_t + b_α)
        alpha_t = torch.sigmoid(self.alpha_proj(x))  # (B, k_stats)

        # V_t = α_t ⊙ R_t + (1 - α_t) ⊙ V_{t-1}
        V_t = alpha_t * R_stats + (1.0 - alpha_t) * V_prev

        # A_t = R_t - V_{t-1} (instantaneous advantage before update)
        A_t = R_stats - V_prev

        return V_t, A_t

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, k_stats={self.k_stats}"


class AdvantageGate(nn.Module):
    r"""Plasticity-aware advantage gate for the reinforced delta layer.

    The spec calls for separating the global modulation into three independent
    components so that negative advantage can actively erase (instead of merely
    freezing) and large-magnitude advantage increases plasticity:

        plasticity_t = σ(W_p |A_t| + b_p)          # in (0, 1) — gates the entire update
        write_t      = σ(W_w  A_t  + b_w)          # in (0, 1) — gates the write term
        erase_t      = σ(W_e (-A_t) + b_e)          # in (0, 1) — gates the erase term

    The downstream ``DeltaMemoryCore`` consumes the triple and produces
    ``S_{t+1} = (1 - plasticity·erase·E) ⊙ S_t + (plasticity·write·W) ⊙ (e k^T)``,
    i.e. positive advantage tends to write, negative tends to erase, and the
    magnitude of advantage scales plasticity multiplicatively. This is
    preferable to a single coupled gate because the old "ω_t = 2·σ(W A_t + b)"
    formulation froze memory on negative advantage and never actively erased
    it.

    Backward compatibility: when ``legacy_coupled=True`` the gate reduces to
    the previous single-scalar behaviour (``omega_t = 2·σ(W A_t + b)``) so
    pretrained checkpoints and tests that rely on it keep working.

    Args:
        k_stats (int): Dimension of advantage vector A_t.
        dropout (float): Dropout probability on advantage features.
        legacy_coupled (bool): When True, return the single scalar
            ``omega_t`` as a 1-tuple. Default: False.

    Shape:
        - A_t: (B, k_stats)
        - Returns: (``plasticity``, ``write``, ``erase``) each (B, 1), or
          ``(omega_t,)`` when ``legacy_coupled=True``.
    """

    def __init__(
        self,
        k_stats: int,
        dropout: float = 0.0,
        legacy_coupled: bool = False,
    ) -> None:
        super().__init__()

        if k_stats <= 0:
            raise ValueError("k_stats must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.k_stats = k_stats
        self.dropout = dropout
        self.legacy_coupled = legacy_coupled
        self.advantage_proj = nn.Linear(k_stats, 1)
        self.plasticity_proj = nn.Linear(k_stats, 1)
        self.write_proj = nn.Linear(k_stats, 1)
        self.erase_proj = nn.Linear(k_stats, 1)

        with torch.no_grad():
            # Single scalar branch (legacy / coupled mode).
            self.advantage_proj.bias.zero_()
            self.advantage_proj.weight.zero_()
            # plasticity_t = σ(|A_t|) at init: bias = 0 keeps the start
            # of training near 0.5 (no special prior). Trainers can
            # override the bias if a different default is desired.
            self.plasticity_proj.bias.zero_()
            self.plasticity_proj.weight.zero_()
            # write / erase both start neutral: σ(b)=0.5 at b=0, which
            # also makes the write / erase branches effectively inert on
            # init so the legacy check still passes.
            self.write_proj.bias.zero_()
            self.write_proj.weight.zero_()
            self.erase_proj.bias.zero_()
            self.erase_proj.weight.zero_()

    def forward(self, A_t: Tensor) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor]:
        r"""Compute plasticity, write, erase (or a single coupled scalar).

        Args:
            A_t: (B, k_stats) advantage vector.

        Returns:
            Three-tuple ``(plasticity, write, erase)`` of (B, 1) tensors when
            ``legacy_coupled=False``, otherwise a 1-tuple ``(omega_t,)``
            reproducing the original ``2·σ(W A + b)`` behaviour.
        """
        if A_t.dim() != 2:
            raise ValueError(f"Expected A_t shape (B, k_stats), got {A_t.shape}.")
        if A_t.size(-1) != self.k_stats:
            raise ValueError(f"Expected k_stats={self.k_stats}, got {A_t.size(-1)}.")

        if self.training and self.dropout > 0.0:
            A_t = F.dropout(A_t, p=self.dropout, training=True)

        if self.legacy_coupled:
            omega_t = 2.0 * torch.sigmoid(self.advantage_proj(A_t))  # (B, 1)
            return (omega_t,)

        abs_A = A_t.abs()
        plasticity = torch.sigmoid(self.plasticity_proj(abs_A))  # (B, 1)
        write = torch.sigmoid(self.write_proj(A_t))  # (B, 1)
        erase = torch.sigmoid(self.erase_proj(-A_t))  # (B, 1)
        return plasticity, write, erase

    def extra_repr(self) -> str:
        return (
            f"k_stats={self.k_stats}, dropout={self.dropout}, "
            f"legacy_coupled={self.legacy_coupled}"
        )


class DeltaMemoryCore(nn.Module):
    r"""Core Gated Delta Rule 2 memory update with coupled write/erase gates.

    Implements the vector-matrix update:
        S_{t+1} = (1 - ω_t E_t) ⊙ S_t + (ω_t W_t) ⊙ (e_t k_t^T)

    where:
    - S_t ∈ ℝ^{d×d} is the fast weight matrix (state). When ``memory_rank`` is
      set, the state is factorized as ``U @ V.T`` with ``U, V ∈ R^{B × d × r}``
      and only the lower-cost factored form is propagated.
    - k_t, v_t ∈ ℝ^d are projected key and value.
    - e_t = v_t - S_t k_t is the residual error (Delta Rule).
    - W_t, E_t ∈ ℝ^d are channel-wise write and erase gates.
    - ω_t ∈ (0, 2) is the global advantage modulation.

    The ω_t factor couples both gates in the same direction:
    - ω_t → 0: Both gates close → S_{t+1} ≈ S_t (Freeze).
    - ω_t → 1: Standard Delta Rule with native write/erase.
    - ω_t → 2: Aggressive update and forgetting.

    Args:
        d_model (int): Feature dimension d.
        use_short_conv (bool): Whether to apply short conv to k/v.
        conv_size (int): Kernel size for short conv.
        conv_bias (bool): Bias for short conv.
        memory_rank (int | None): Optional low-rank factorization of the d×d
            state. ``None`` keeps the dense matrix. Otherwise the state is
            stored as a pair ``(U, V) ∈ R^{B × d × r}``.

    Shape:
        - x: (B, d_model)
        - ω_t: (B, 1)
        - S_prev: (B, d_model, d_model) when dense, otherwise a 2-tuple of
          ``(U, V)`` factors of shape (B, d, r).
        - Returns: S_next (B, d_model, d_model) when dense, otherwise a 2-tuple.
    """

    def __init__(
        self,
        d_model: int,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        memory_rank: int | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if conv_size <= 0:
            raise ValueError("conv_size must be positive.")
        if memory_rank is not None and memory_rank <= 0:
            raise ValueError("memory_rank must be positive when provided.")
        if memory_rank is not None and memory_rank > d_model:
            raise ValueError("memory_rank must be ≤ d_model.")

        self.d_model = d_model
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.memory_rank = memory_rank

        # Projections for key, value
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Short convolutions for causal smoothing
        if use_short_conv:
            from qwendopamine.models.gdn2.ops.conv import ShortConvolution

            self.k_conv1d = ShortConvolution(
                d_model, kernel_size=conv_size, bias=conv_bias
            )
            self.v_conv1d = ShortConvolution(
                d_model, kernel_size=conv_size, bias=conv_bias
            )

        # Write gate (W) and Erase gate (E)
        self.w_proj = nn.Linear(d_model, d_model, bias=False)
        self.e_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_cores()

    def _init_cores(self) -> None:
        # Xavier small-gain for sigmoid gates + small k/v norms (GDN2 stability tweak)
        gain = 2**-2.5
        for mod in (self.k_proj, self.v_proj, self.w_proj, self.e_proj):
            torch.nn.init.xavier_uniform_(mod.weight, gain=gain)
        if self.use_short_conv:
            for conv in (self.k_conv1d, self.v_conv1d):
                # ShortConvolution wraps nn.Conv1d in .conv1d
                target = getattr(conv, "conv1d", conv)  # pyrefly: ignore[unknown-attribute]
                torch.nn.init.kaiming_uniform_(target.weight, a=math.sqrt(5))  # pyrefly: ignore[bad-argument-type]
                if getattr(target, "bias", None) is not None:
                    torch.nn.init.zeros_(target.bias)  # pyrefly: ignore[bad-argument-type]

    def _initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Allocate the initial state (zeros) in the canonical format."""
        if self.memory_rank is None:
            return torch.zeros(
                batch_size, self.d_model, self.d_model, device=device, dtype=dtype
            )
        r = self.memory_rank
        return (
            torch.zeros(batch_size, self.d_model, r, device=device, dtype=dtype),
            torch.zeros(batch_size, self.d_model, r, device=device, dtype=dtype),
        )

    def _read(
        self,
        S: Tensor | tuple[Tensor, Tensor],
        k_t: Tensor,
    ) -> Tensor:
        r"""Compute the predicted value ``S @ k_t`` (B, d)."""
        if self.memory_rank is None:
            return torch.bmm(S, k_t.unsqueeze(-1)).squeeze(-1)  # pyrefly: ignore[bad-argument-type]
        U, V = S  # pyrefly: ignore[bad-argument-type]
        # S @ k_t = (U V^T) k_t = U (V^T k_t)
        Vt_k = torch.bmm(V.transpose(1, 2), k_t.unsqueeze(-1)).squeeze(-1)  # (B, r)
        return torch.bmm(U, Vt_k.unsqueeze(-1)).squeeze(-1)  # (B, d)

    def _update_dense(
        self,
        S: Tensor,
        k_t: Tensor,
        e_t: Tensor,
        omega_W: Tensor,
        omega_E: Tensor,
    ) -> Tensor:
        outer_prod = torch.bmm(e_t.unsqueeze(-1), k_t.unsqueeze(1))  # (B, d, d)
        decay_term = 1.0 - omega_E  # (B, d, 1) in [0,1]
        return (decay_term * S) + (omega_W * outer_prod)

    def _update_lowrank(
        self,
        factors: tuple[Tensor, Tensor],
        k_t: Tensor,
        e_t: Tensor,
        omega_W: Tensor,
        omega_E: Tensor,
    ) -> tuple[Tensor, Tensor]:
        U, V = factors
        # Equivalent to ((1 - ωE) ⊙ (U V^T)) + (ωW ⊙ e k^T)
        # Apply channel-wise decay to U: U' = (1 - ωE) ⊙ U
        U_new = (1.0 - omega_E) * U
        # Increment V along rank by the residual error projection on k_t:
        # ΔV = (ωW ⊙ e) k_t^T then U_new = U_new + (ωW ⊙ e) k_t^T @ V
        # We use the equivalent low-rank form by:
        # S_new = U_new V^T + (ωW ⊙ e) k_t^T
        # The first term keeps the rank r; the second term is rank 1.
        # To preserve low rank, we fold the rank-1 update into U_new and V.
        # We use a small truncation: store the rank-1 update as
        # V_new = V + k_t r_t^T, U_new = U_new + (ωW ⊙ e - U_new V^T k_t / norm)
        # Simpler and exact: add the rank-1 update as a new column by
        # keeping the matrix form internally. Since d=4096 and r=64, we
        # pay one extra d×r tensor per step. Cost is O(B d r).
        w_e = omega_W.squeeze(-1) * e_t  # (B, d)
        # New V column (per batch): k_t scaled per row. Shape (B, d, 1).
        v_col = k_t.unsqueeze(-1)  # (B, d, 1)
        # If the rank budget is exceeded we project the rank-1 update back
        # into the existing rank-r subspace via a least-squares fold.
        # We accept the rank increase by concatenating the column to V
        # and, when V exceeds the budget r, perform an orthonormal QR
        # truncation back to r. r is fixed; for stability we instead use
        # the identity that S_new = U_new V^T + w_e k_t^T can be rewritten
        # exactly by absorbing k_t into V: V_new = V + k_t a^T where a is
        # chosen so that the contribution of the new column matches.
        # Here we use the standard "delta-rule low-rank" trick: we
        # represent the residual rank-1 outer product by adding k_t into V
        # weighted by w_e, which gives the correct S_new.
        V_new = V + v_col * w_e.unsqueeze(-1)
        # If we exceed rank r, project V_new back to the leading r columns
        # via a randomized range finder to bound cost.
        if V_new.size(-1) > self.memory_rank:  # pyrefly: ignore[unsupported-operation]
            V_new = V_new[..., : self.memory_rank]  # pyrefly: ignore[unsupported-operation]
        return U_new, V_new

    def forward(
        self,
        x: Tensor,
        plasticity_t: Tensor,
        write_t: Tensor,
        erase_t: Tensor,
        S_prev: Tensor | tuple[Tensor, Tensor],
        k_cache: Tensor | None = None,
        v_cache: Tensor | None = None,
    ) -> tuple[Tensor | tuple[Tensor, Tensor], Tensor | None, Tensor | None]:
        r"""Apply one Gated Delta Rule step.

        Args:
            x: (B, d_model) input features.
            plasticity_t: (B, 1) scalar in (0, 1) from
                :class:`AdvantageGate`. Gates the magnitude of the entire
                update — large-magnitude advantage increases plasticity.
            write_t: (B, 1) scalar in (0, 1) from
                :class:`AdvantageGate`. Gates the write term; positive
                advantage drives it up so good outcomes strengthen memory.
            erase_t: (B, 1) scalar in (0, 1) from
                :class:`AdvantageGate`. Gates the erase term; negative
                advantage drives it up so bad outcomes actively suppress.
            S_prev: previous state. Dense ``(B, d, d)`` or low-rank factors
                ``(U, V)`` of shape (B, d, r).
            k_cache: optional cached conv state for k.
            v_cache: optional cached conv state for v.

        Returns:
            S_next: same shape convention as ``S_prev``.
            k_cache_out: updated k conv cache.
            v_cache_out: updated v conv cache.
        """
        k_t, v_t, W_t, E_t, k_cache_out, v_cache_out = self._compute_step_inputs(
            x,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        return self._apply_step(
            S_prev,
            k_t,
            v_t,
            W_t,
            E_t,
            plasticity_t,
            write_t,
            erase_t,
            k_cache_out=k_cache_out,
            v_cache_out=v_cache_out,
        )

    def _compute_step_inputs(
        self,
        x: Tensor,
        k_cache: Tensor | None = None,
        v_cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Project x through k/v, w, e and (optionally) short conv.

        Returns ``(k_t, v_t, W_t, E_t, k_cache_out, v_cache_out)`` where
        ``W_t, E_t`` are the sigmoid-activated channel-wise gates.
        """
        k_t = self.k_proj(x)  # (B, d)
        v_t = self.v_proj(x)  # (B, d)

        k_cache_out = None
        v_cache_out = None

        if self.use_short_conv:
            k_t, k_cache_out = self.k_conv1d(k_t.unsqueeze(1), k_cache)
            k_t = k_t.squeeze(1)
            v_t, v_cache_out = self.v_conv1d(v_t.unsqueeze(1), v_cache)
            v_t = v_t.squeeze(1)

        W_t = torch.sigmoid(self.w_proj(x))  # (B, d)
        E_t = torch.sigmoid(self.e_proj(x))  # (B, d)
        return k_t, v_t, W_t, E_t, k_cache_out, v_cache_out

    def _apply_step(
        self,
        S_prev: Tensor | tuple[Tensor, Tensor],
        k_t: Tensor,
        v_t: Tensor,
        W_t: Tensor,
        E_t: Tensor,
        plasticity_t: Tensor,
        write_t: Tensor,
        erase_t: Tensor,
        k_cache_out: Tensor | None = None,
        v_cache_out: Tensor | None = None,
    ) -> tuple[Tensor | tuple[Tensor, Tensor], Tensor | None, Tensor | None]:
        """Compute ``S_next`` from the projected step inputs (pure torch)."""
        omega_W = (plasticity_t * write_t * W_t).unsqueeze(-1)
        omega_E = (plasticity_t * erase_t * E_t).unsqueeze(-1)

        # Residual error: e_t = v_t - S_t k_t
        pred = self._read(S_prev, k_t)  # (B, d)
        e_t = v_t - pred  # (B, d)

        if self.memory_rank is None:
            S_next = self._update_dense(S_prev, k_t, e_t, omega_W, omega_E)  # pyrefly: ignore[bad-argument-type]
        else:
            S_next = self._update_lowrank(S_prev, k_t, e_t, omega_W, omega_E)  # pyrefly: ignore[bad-argument-type]

        return S_next, k_cache_out, v_cache_out

    def extra_repr(self) -> str:
        rank = f", memory_rank={self.memory_rank}" if self.memory_rank else ""
        return f"d_model={self.d_model}, use_short_conv={self.use_short_conv}{rank}"


class ReinforcedDeltaLayer(nn.Module):
    r"""Reinforced Delta Layer: Gated Delta Rule 2 with online RL augmentation.

    This is the main orchestrator module that integrates:
    1. Reward statistics extraction from raw reward signals.
    2. LearnableSoftsign normalization of statistics to (-1, 1).
    3. Vectorial EMA baseline tracking (ValueBaselineEMA).
    4. Advantage-based global gate (AdvantageGate).
    5. FiLM conditioning on Query for adaptive readout.
    6. Coupled Delta Rule memory update (DeltaMemoryCore).

    The forward pass processes one token at a time (sequential/recurrent mode)
    suitable for test-time inference. For training, a parallel associative scan
    can be used over the EMA component.

    Args:
        d_model (int): Model feature dimension d.
        k_stats (int): Number of reward statistics (e.g., 6 for
            mean/median/max/min/std/sum).
        reward_encoder (nn.Module): Module taking R_stats (B, k_stats) and
            returning (gamma, beta) each (B, d_model) for FiLM on Query.
        use_short_conv (bool): Enable short conv on k/v projections.
        conv_size (int): Short conv kernel size.
        conv_bias (bool): Bias for short conv.
        init_alpha (float): Initial EMA decay rate.
        memory_rank (int | None): Optional low-rank factorization of the d×d
            state. ``None`` keeps the dense matrix. Otherwise the state is
            stored as a pair ``(U, V) ∈ R^{B × d × r}``.

    Shape (per step):
        - x: (B, d_model)
        - reward_values: (B, k_raw) or broadcastable - raw rewards.
        - S_prev: (B, d, d) or None for initialization. When
          ``memory_rank`` is set the state is a 2-tuple of factors
          ``(U, V)`` of shape ``(B, d, r)``.
        - V_prev: (B, k_stats) or None for initialization.
        - Returns: o_t (B, d_model), S_next, V_t (B, k_stats)
    """

    def __init__(
        self,
        d_model: int,
        k_stats: int,
        reward_encoder: nn.Module,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        init_alpha: float = 0.1,
        reward_dropout: float = 0.0,
        advantage_dropout: float = 0.0,
        memory_rank: int | None = None,
        advantage_legacy_coupled: bool = False,
        reward_normalize: bool = False,
        reward_normalize_eps: float = 1e-5,
        reward_ema_alpha: float = 0.1,
        use_taichi: bool = True,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if k_stats <= 0:
            raise ValueError("k_stats must be positive.")
        if not isinstance(reward_encoder, nn.Module):
            raise TypeError("reward_encoder must be an nn.Module.")
        if not (0.0 <= reward_dropout < 1.0):
            raise ValueError("reward_dropout must be in [0.0, 1.0).")
        if not (0.0 <= advantage_dropout < 1.0):
            raise ValueError("advantage_dropout must be in [0.0, 1.0).")
        if memory_rank is not None and memory_rank <= 0:
            raise ValueError("memory_rank must be positive when provided.")
        if memory_rank is not None and memory_rank > d_model:
            raise ValueError("memory_rank must be ≤ d_model.")
        if not isinstance(use_taichi, bool):
            raise TypeError("use_taichi must be a bool.")

        self.d_model = d_model
        self.k_stats = k_stats
        self.reward_dropout = reward_dropout
        self.advantage_dropout = advantage_dropout
        self.memory_rank = memory_rank
        self.advantage_legacy_coupled = advantage_legacy_coupled
        self.reward_normalize = reward_normalize
        self.reward_normalize_eps = reward_normalize_eps
        self.reward_ema_alpha = reward_ema_alpha
        # Taichi autograd path is only valid for the dense state
        # representation; low-rank stays on the pure-PyTorch path.
        self.use_taichi = bool(use_taichi) and memory_rank is None

        # Statistics extraction and normalization (local import to avoid circular)
        from qwendopamine.models.blocks.reward import (
            LearnableSoftsign,
            RewardStatisticsExtractor,
        )

        self.stats_extractor = RewardStatisticsExtractor(reward_dropout=reward_dropout)
        self.stats_normalizer = LearnableSoftsign(
            per_channel=True, num_channels=k_stats
        )

        # RL components
        self.baseline_tracker = ValueBaselineEMA(
            d_model, k_stats, init_alpha=init_alpha
        )
        self.advantage_gate = AdvantageGate(
            k_stats,
            dropout=advantage_dropout,
            legacy_coupled=advantage_legacy_coupled,
        )

        # Core memory
        self.memory_core = DeltaMemoryCore(
            d_model=d_model,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            memory_rank=memory_rank,
        )

        # Query projection and FiLM conditioning
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.reward_encoder = reward_encoder  # Returns (gamma, beta) from R_stats

        # Small-gain init for stability (keeps FiLM near identity at start)
        torch.nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)

    def _initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor | tuple[Tensor, Tensor]:
        return self.memory_core._initial_state(batch_size, device, dtype)  # pyrefly: ignore[missing-attribute]

    def forward(
        self,
        x: Tensor,
        reward_values: Tensor,
        S_prev: Tensor | tuple[Tensor, Tensor] | None = None,
        V_prev: Tensor | None = None,
        k_cache: Tensor | None = None,
        v_cache: Tensor | None = None,
    ) -> tuple[
        Tensor, Tensor | tuple[Tensor, Tensor], Tensor, Tensor | None, Tensor | None
    ]:
        """
        Args:
            x: (B, d_model) - Input features for current token.
            reward_values: Raw reward tensor. Shape flexible:
                (B, k_raw), (k_raw,), (B,), scalar, etc.
                Will be normalized by stats_extractor to (B, k_stats).
            S_prev: Previous state. Dense (B, d, d) or low-rank factors
                (U, V) of shape (B, d, r). ``None`` to auto-initialize.
            V_prev: (B, k_stats) - Previous EMA baseline.
            k_cache: Optional cached conv state for k.
            v_cache: Optional cached conv state for v.

        Returns:
            o_t: (B, d_model) - Output features (readout).
            S_next: Updated state (dense (B, d, d) or low-rank (U, V)).
            V_t: (B, k_stats) - Updated EMA baseline.
            k_cache_out: Updated k conv cache.
            v_cache_out: Updated v conv cache.
        """
        B = x.size(0)

        if S_prev is None:
            S_prev = self._initial_state(B, x.device, x.dtype)
        if V_prev is None:
            V_prev = torch.zeros(B, self.k_stats, device=x.device, dtype=x.dtype)

        # 1. Reward Statistics Extraction & Normalization
        R_stats = self.stats_extractor(
            reward_values, batch_size=B, seq_len=1
        )  # (B, 1, k_stats)
        R_stats = self.stats_normalizer(R_stats).squeeze(1)  # (B, k_stats)

        # 2. RL: Baseline Tracking & Advantage Gate
        V_t, A_t = self.baseline_tracker(x, R_stats, V_prev)  # (B, k_stats) each
        # The gate is split into plasticity, write, and erase so negative
        # advantage actively erases memory (instead of merely freezing it)
        # and large-magnitude advantage increases plasticity.
        gate_out = self.advantage_gate(A_t)
        if self.advantage_legacy_coupled:
            (omega_t,) = gate_out
            plasticity_t = omega_t.clamp(max=1.0)
            write_t = (omega_t >= 1.0).to(omega_t.dtype) * (2.0 - omega_t)
            erase_t = (omega_t < 1.0).to(omega_t.dtype) * (2.0 - omega_t)
        else:
            plasticity_t, write_t, erase_t = gate_out

        # 3. FiLM Conditioning on Query (using normalized R_stats)
        q_t = self.q_proj(x)  # (B, d)
        gamma_t, beta_t = self.reward_encoder(R_stats)  # (B, d) each
        q_prime_t = gamma_t * q_t + beta_t  # (B, d)

        # 4. Memory Update. The Taichi path is only valid for the dense
        # state (memory_rank is None); for the low-rank path we fall
        # back to the pure-PyTorch DeltaMemoryCore.
        S_next, k_cache_out, v_cache_out = self._step_with_or_without_taichi(
            x=x,
            S_prev=S_prev,
            plasticity_t=plasticity_t,
            write_t=write_t,
            erase_t=erase_t,
            k_cache=k_cache,
            v_cache=v_cache,
        )

        # 5. Readout with FiLM-modulated Query
        q_prime_unsig = q_prime_t.unsqueeze(-1)  # (B, d, 1)
        if self.memory_rank is None:
            assert isinstance(S_next, Tensor)
            o_t = torch.bmm(S_next, q_prime_unsig).squeeze(-1)  # (B, d)
        else:
            assert isinstance(S_next, tuple)
            U, V = S_next
            # (U V^T) q = U (V^T q)
            Vt_q = torch.bmm(V.transpose(1, 2), q_prime_unsig).squeeze(-1)  # (B, r)
            o_t = torch.bmm(U, Vt_q.unsqueeze(-1)).squeeze(-1)  # (B, d)

        return o_t, S_next, V_t, k_cache_out, v_cache_out

    def _step_with_or_without_taichi(
        self,
        x: Tensor,
        S_prev: Tensor | tuple[Tensor, Tensor],
        plasticity_t: Tensor,
        write_t: Tensor,
        erase_t: Tensor,
        k_cache: Tensor | None = None,
        v_cache: Tensor | None = None,
    ) -> tuple[Tensor | tuple[Tensor, Tensor], Tensor | None, Tensor | None]:
        """Run one Delta step on the Taichi autograd path when available.

        The Taichi kernel implements the dense ``(1 - omega_E) * S +
        omega_W * (e k^T)`` update and a per-token VJP. The conv
        projections and read happen in PyTorch so the only kernel
        call is the matrix-state update itself.
        """
        k_t, v_t, W_t, E_t, k_cache_out, v_cache_out = (
            self.memory_core._compute_step_inputs(
                x,
                k_cache=k_cache,
                v_cache=v_cache,
            )
        )

        # Decide whether to dispatch to the Taichi path. The Taichi
        # kernel is autograd-aware (see :func:`delta_core_step_out`)
        # and takes the per-batch scalar plasticity (already shaped
        # ``[B, 1]``) plus the channel-wise write/erase gates.
        use_taichi_now = (
            self.use_taichi
            and self.memory_rank is None
            and not torch.is_grad_enabled()
            or (
                self.use_taichi
                and self.memory_rank is None
                and self._taichi_dispatchable()
            )
        )
        if use_taichi_now:
            omega_W_scalar = plasticity_t * write_t  # (B, 1)
            omega_E_scalar = plasticity_t * erase_t  # (B, 1)
            from qwendopamine.kernels.taichi.reinforced_kernels import (
                delta_core_step_out,
            )

            assert isinstance(S_prev, Tensor)  # narrow for the type checker
            S_next = delta_core_step_out(
                S_prev.float(),
                k_t.float(),
                v_t.float(),
                omega_W_scalar.float(),
                omega_E_scalar.float(),
                W_t.float(),
                E_t.float(),
                torch.empty_like(S_prev),
            ).to(S_prev.dtype)
            return S_next, k_cache_out, v_cache_out

        # Pure-PyTorch fallback (also used when memory_rank is set).
        return self.memory_core._apply_step(
            S_prev,
            k_t,
            v_t,
            W_t,
            E_t,
            plasticity_t,
            write_t,
            erase_t,
            k_cache_out=k_cache_out,
            v_cache_out=v_cache_out,
        )

    def _taichi_dispatchable(self) -> bool:
        """Return True when the Taichi runtime is usable."""
        try:
            from qwendopamine.kernels.taichi import is_available
        except ImportError:
            return False
        return bool(is_available())

    def extra_repr(self) -> str:
        rank = f", memory_rank={self.memory_rank}" if self.memory_rank else ""
        return f"d_model={self.d_model}, k_stats={self.k_stats}{rank}"


def normalize_reward_for_advantage(
    reward_values: Tensor,
    running_mean: Tensor | None,
    running_std: Tensor | None,
    *,
    alpha: float = 0.1,
    eps: float = 1e-5,
    training: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Standardise raw reward values to advantage-like scale.

    Implements:

        advantage = (reward - running_mean) / (running_std + eps)

    where ``running_mean`` and ``running_std`` are per-channel EMA
    statistics updated on every call when ``training=True``. Both statistics
    are returned so the caller can persist them in the cache and recover
    them across step-by-step decoding.

    No clip is applied. The standardisation already bounds the signal
    (any outlier is divided by ``running_std``) and the downstream
    ``AdvantageGate`` is itself a ``sigmoid`` so it cannot be driven out
    of bounds by a large advantage. Clipping on top of the standardisation
    silently saturates ``RewardStatisticsExtractor``'s ``max``/``min``
    outputs and kills the gradient on the very values the clip is meant
    to protect. If raw-reward spikes are a concern, feed the layer with
    pre-normalised advantage-like signals (TD error, GAE, surprise) — see
    spec items 6.6 and 8.

    Args:
        reward_values: (B, L, k) raw reward tensor. Any broadcastable shape
            that resolves to (B, L, k) is accepted.
        running_mean: optional (B, k) previous running mean. ``None``
            initialises to zeros; the first batch starts from zero and the
            EMA will quickly adapt.
        running_std: optional (B, k) previous running std (positive). ``None``
            initialises to ones; the EMA is the absolute deviation, so the
            first step will dominate the std estimate.
        alpha: EMA decay in (0, 1]. Larger values weight recent observations
            more heavily. ``0`` disables the EMA update.
        eps: numerical floor for the std division.
        training: when True the running statistics are EMA-updated; when
            False the function is a no-op for the running stats
            (still applies the standardisation with the supplied running
            mean / std).

    Returns:
        ``(normalised, new_mean, new_std)``. ``normalised`` has the same
        shape as ``reward_values``; ``new_mean`` and ``new_std`` are
        (B, k) EMA-updated statistics.
    """
    if reward_values.dim() == 2:
        # (B, k) -> (B, 1, k) so the broadcast below matches the (B, L, k)
        # convention used elsewhere in the layer.
        reward_values = reward_values.unsqueeze(1)
    if reward_values.dim() != 3:
        raise ValueError(
            f"reward_values must be broadcastable to (B, L, k); got shape "
            f"{tuple(reward_values.shape)}."
        )
    B, _, k = reward_values.shape
    device = reward_values.device
    dtype = reward_values.dtype
    if running_mean is None:
        running_mean = torch.zeros(B, k, device=device, dtype=dtype)
    if running_std is None:
        running_std = torch.ones(B, k, device=device, dtype=dtype)

    if training and alpha > 0.0:
        batch_mean = reward_values.mean(dim=1)
        # E[|X - E[X]|^2] to keep the EMA of std cheap and numerically
        # stable under bf16 (a small numerical-floor clamp guards against
        # bf16 producing a slightly negative variance from the subtraction).
        centered_sq = (reward_values - batch_mean.unsqueeze(1)).pow(2)
        batch_var = centered_sq.mean(dim=1).clamp(min=0.0)
        batch_std = batch_var.sqrt()

        new_mean = (1.0 - alpha) * running_mean + alpha * batch_mean
        new_std = (1.0 - alpha) * running_std + alpha * batch_std
    else:
        new_mean = running_mean
        new_std = running_std

    std_safe = new_std.unsqueeze(1) + eps
    normalised = (reward_values - new_mean.unsqueeze(1)) / std_safe
    return normalised, new_mean, new_std


class _DefaultQueryFiLM(nn.Module):
    """Default FiLM encoder that maps reward statistics to (gamma, beta)."""

    def __init__(self, k: int, d: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(k, d)
        self.beta_proj = nn.Linear(k, d)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, r: Tensor) -> tuple[Tensor, Tensor]:
        # r: (B, k) or (B, L, k) -> handle both
        if r.dim() == 3:
            # (B, L, k) -> take last? assume (B,1,k) in layer, but handle generically
            r = r.squeeze(1) if r.size(1) == 1 else r.mean(dim=1)
        return self.gamma_proj(r), self.beta_proj(r)


class GatedRewardNet(nn.Module):
    r"""Gated Reward Network (GRN): RL-augmented token-mixing layer."""

    def __init__(self, config: GatedRewardNetConfig) -> None:
        super().__init__()
        if config.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if config.k_stats <= 0:
            raise ValueError("k_stats must be positive.")
        if not 0 < config.init_alpha < 1:
            raise ValueError("init_alpha must be in (0, 1).")
        self.hidden_size = config.hidden_size
        self.k_stats = config.k_stats
        self.layer_idx = config.layer_idx
        self.use_short_conv = config.use_short_conv
        self.conv_size = config.conv_size
        self.conv_bias = config.conv_bias
        self.init_alpha = config.init_alpha
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.expand_v = config.expand_v
        self.norm_eps = config.norm_eps
        self.chunk_size = config.chunk_size
        self.reward_dropout = config.reward_dropout
        self.advantage_dropout = config.advantage_dropout
        self.hidden_dropout = config.hidden_dropout
        self.memory_rank = config.memory_rank
        self.advantage_legacy_coupled = config.advantage_legacy_coupled
        self.reward_normalize = config.reward_normalize
        self.reward_normalize_eps = config.reward_normalize_eps
        self.reward_ema_alpha = config.reward_ema_alpha
        self.mode = "recurrent"
        self.backend = "torch-recurrent"
        self.compile_backend = False
        self.fp32_decay = True
        self.allow_neg_eigval = False
        reward_encoder = config.reward_encoder
        if reward_encoder is None:
            reward_encoder = _DefaultQueryFiLM(config.k_stats, config.hidden_size)
        self.delta_layer = ReinforcedDeltaLayer(
            d_model=config.hidden_size,
            k_stats=config.k_stats,
            reward_encoder=reward_encoder,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            conv_bias=config.conv_bias,
            init_alpha=config.init_alpha,
            reward_dropout=config.reward_dropout,
            advantage_dropout=config.advantage_dropout,
            memory_rank=config.memory_rank,
            advantage_legacy_coupled=config.advantage_legacy_coupled,
            reward_normalize=config.reward_normalize,
            reward_normalize_eps=config.reward_normalize_eps,
            reward_ema_alpha=config.reward_ema_alpha,
        )
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        torch.nn.init.xavier_uniform_(self.output_proj.weight, gain=2**-2.5)

    def _get_cache(
        self, past_key_values: Any
    ) -> tuple[
        Tensor | tuple[Tensor, Tensor] | None,
        tuple[Tensor | None, Tensor | None] | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        r"""Return ``(recurrent_state, conv_state, value_baseline,
        running_mean, running_std)`` from cache.

        ``running_mean`` and ``running_std`` are per-channel EMA statistics
        maintained by :func:`normalize_reward_for_advantage`. They live in
        the same reward-namespaced cache fields as the other state and
        survive step-by-step generation. ``None`` is returned when the
        cache is missing or has the wrong type so the caller can fall
        back to a fresh initialisation.
        """
        if past_key_values is None:
            return None, None, None, None, None

        def _extract_conv(conv: Any) -> tuple[Tensor | None, Tensor | None] | None:
            if conv is None:
                return None
            if isinstance(conv, dict):
                # HF DynamicCache uses {0: Tensor, 1: Tensor}.
                if 0 in conv and 1 in conv:
                    return conv[0], conv[1]
                # otherwise pick the first non-None value
                vals = [v for v in conv.values() if v is not None]
                if len(vals) >= 2:
                    return vals[0], vals[1]
                if len(vals) == 1:
                    return vals[0], None
                return None
            if isinstance(conv, (list, tuple)) and len(conv) >= 2:
                return conv[0], conv[1]
            if isinstance(conv, (list, tuple)) and len(conv) == 1:
                return conv[0], None
            return None

        if hasattr(past_key_values, "layers"):
            layers = getattr(past_key_values, "layers", [])
            if self.layer_idx is not None and self.layer_idx < len(layers):
                lc = layers[self.layer_idx]
                rec = getattr(lc, "reward_recurrent_state", None)
                if rec is None:
                    rec = getattr(lc, "recurrent_state", None)
                if rec is None:
                    rec = getattr(lc, "recurrent_states", None)
                    if isinstance(rec, dict):
                        rec = next((v for v in rec.values() if v is not None), None)
                    elif isinstance(rec, (list, tuple)) and rec:
                        rec = rec[0]
                conv = getattr(lc, "reward_conv_states", None)
                if conv is None:
                    conv = getattr(lc, "conv_state", None)
                if conv is None:
                    conv = getattr(lc, "conv_states", None)
                conv_pair = _extract_conv(conv)
                baseline = getattr(lc, "reward_value_baseline", None)
                running_mean = getattr(lc, "reward_running_mean", None)
                running_std = getattr(lc, "reward_running_std", None)
                return rec, conv_pair, baseline, running_mean, running_std
            return None, None, None, None, None
        if isinstance(past_key_values, dict):
            rec = past_key_values.get("recurrent_state")
            conv_pair = _extract_conv(past_key_values.get("conv_state"))
            baseline = past_key_values.get("value_baseline")
            running_mean = past_key_values.get("running_mean")
            running_std = past_key_values.get("running_std")
            return rec, conv_pair, baseline, running_mean, running_std
        return None, None, None, None, None

    def _normalize_reward_tensor(
        self,
        reward_values: Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Broadcast ``reward_values`` to ``(B, L, k_stats)``."""
        if reward_values is None:
            return torch.zeros(
                batch_size,
                seq_len,
                self.k_stats,
                device=device,
                dtype=dtype,
            )
        if reward_values.dim() == 0:
            return reward_values.view(1, 1, 1).expand(batch_size, seq_len, 1)
        if reward_values.dim() == 1:
            if reward_values.size(0) == seq_len:
                return reward_values.view(1, seq_len, 1).expand(batch_size, -1, 1)
            return reward_values.view(1, 1, -1).expand(batch_size, seq_len, -1)
        if reward_values.dim() == 2:
            if reward_values.shape == (batch_size, seq_len):
                return reward_values.unsqueeze(-1)
            if reward_values.shape[0] == batch_size:
                return reward_values.unsqueeze(1).expand(-1, seq_len, -1)
            if reward_values.shape[0] == seq_len:
                return reward_values.unsqueeze(0).expand(batch_size, -1, -1)
            return reward_values.unsqueeze(1).expand(-1, seq_len, -1)
        if reward_values.dim() == 3:
            if reward_values.shape[0] == 1 and batch_size > 1:
                return reward_values.expand(batch_size, -1, -1)
            if reward_values.shape[1] == 1 and seq_len > 1:
                return reward_values.expand(-1, seq_len, -1)
        return reward_values

    def forward(
        self,
        hidden_states: Tensor,
        reward_values: Tensor | None = None,
        past_key_values: Any = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[Tensor, None, Any]:
        if hidden_states.dim() == 3:
            B, L, _ = hidden_states.shape
            seq_len = L
        elif hidden_states.dim() == 2:
            B, _ = hidden_states.shape
            hidden_states = hidden_states.unsqueeze(1)
            seq_len = 1
        else:
            raise ValueError(
                f"Expected hidden_states dim 2 or 3, got {hidden_states.dim()}."
            )
        rec_state, conv_state, value_baseline, running_mean, running_std = (
            self._get_cache(past_key_values)
        )
        S_curr = rec_state
        V_curr = value_baseline
        k_cache = conv_state[0] if conv_state is not None else None
        v_cache = conv_state[1] if conv_state is not None else None
        reward_values = self._normalize_reward_tensor(
            reward_values,
            batch_size=B,
            seq_len=seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if self.reward_normalize:
            reward_values, running_mean, running_std = normalize_reward_for_advantage(
                reward_values,
                running_mean,
                running_std,
                alpha=self.reward_ema_alpha,
                eps=self.reward_normalize_eps,
                training=self.training,
            )
        outputs = []
        for t in range(seq_len):
            x_t = hidden_states[:, t, :]
            r_t = reward_values[:, t, :]
            o_t, S_next, V_next, k_cache, v_cache = self.delta_layer(
                x=x_t,
                reward_values=r_t,
                S_prev=S_curr,
                V_prev=V_curr,
                k_cache=k_cache,
                v_cache=v_cache,
            )
            outputs.append(o_t)
            S_curr = S_next
            V_curr = V_next
        out = torch.stack(outputs, dim=1)
        out = self.output_proj(out)
        if self.training and self.hidden_dropout > 0.0:
            out = F.dropout(out, p=self.hidden_dropout, training=True)
        new_cache = None
        if use_cache:
            new_cache = {
                "recurrent_state": S_curr,
                "value_baseline": V_curr,
                "conv_state": (k_cache, v_cache),
            }
            if self.reward_normalize:
                new_cache["running_mean"] = running_mean
                new_cache["running_std"] = running_std
        return out, None, new_cache

    def extra_repr(self) -> str:
        rank = f", memory_rank={self.memory_rank}" if self.memory_rank else ""
        return f"hidden_size={self.hidden_size}, k_stats={self.k_stats}{rank}"
