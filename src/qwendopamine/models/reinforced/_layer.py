# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

"""Reinforced Delta Layer and reward normalisation helper."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ._advantage_gate import AdvantageGate
from ._memory_core import DeltaMemoryCore
from ._value_baseline import ValueBaselineEMA

__all__ = ["ReinforcedDeltaLayer", "normalize_reward_for_advantage"]


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
          ``(U, V)`` of shape (B, d, r).
        - V_prev: (B, k_stats) or None for initialization.
        - Returns: o_t (B, d_model), S_next, V_t (B, k_stats)

    Dtype contract:
        The recurrent state ``S`` is carried in the caller's input
        dtype on every public boundary. The Taichi kernel and the
        PyTorch reference both upcast to float32 internally for the
        per-step update, then cast back. Callers therefore observe
        ``S_next.dtype == S_prev.dtype`` even though the kernel
        computes in float32. This is independent of the
        ``fp32_decay`` knob (which only affects the GDN-2 path).
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
        self.baseline_tracker = ValueBaselineEMA(
            d_model, k_stats, init_alpha=init_alpha
        )
        self.advantage_gate = AdvantageGate(
            k_stats,
            dropout=advantage_dropout,
            legacy_coupled=advantage_legacy_coupled,
        )
        self.memory_core = DeltaMemoryCore(
            d_model=d_model,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            memory_rank=memory_rank,
        )
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.reward_encoder = reward_encoder

        # Small-gain init keeps FiLM near identity at start.
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

        R_stats = self.stats_extractor(
            reward_values, batch_size=B, seq_len=1
        )  # (B, 1, k_stats)
        R_stats = self.stats_normalizer(R_stats).squeeze(1)  # (B, k_stats)

        V_t, A_t = self.baseline_tracker(x, R_stats, V_prev)  # (B, k_stats) each
        gate_out = self.advantage_gate(A_t)
        if self.advantage_legacy_coupled:
            (omega_t,) = gate_out
            plasticity_t = omega_t.clamp(max=1.0)
            write_t = (omega_t >= 1.0).to(omega_t.dtype) * (2.0 - omega_t)
            erase_t = (omega_t < 1.0).to(omega_t.dtype) * (2.0 - omega_t)
        else:
            plasticity_t, write_t, erase_t = gate_out

        q_t = self.q_proj(x)  # (B, d)
        gamma_t, beta_t = self.reward_encoder(R_stats)  # (B, d) each
        q_prime_t = gamma_t * q_t + beta_t  # (B, d)

        # Taichi path is only valid for the dense state (memory_rank is None);
        # low-rank falls back to the pure-PyTorch DeltaMemoryCore.
        S_next, k_cache_out, v_cache_out = self._step_with_or_without_taichi(
            x=x,
            S_prev=S_prev,
            plasticity_t=plasticity_t,
            write_t=write_t,
            erase_t=erase_t,
            k_cache=k_cache,
            v_cache=v_cache,
        )

        q_prime_unsig = q_prime_t.unsqueeze(-1)  # (B, d, 1)
        if self.memory_rank is None:
            assert isinstance(S_next, Tensor)
            o_t = torch.bmm(S_next, q_prime_unsig).squeeze(-1)  # (B, d)
        else:
            assert isinstance(S_next, tuple)
            U, V = S_next
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
        # ``[B, 1]``) plus the channel-wise write/erase gates. We use
        # Taichi when the runtime is available and the model was
        # configured to use it, regardless of grad mode (the kernel
        # records the per-token adjoint when needed).
        use_taichi_now = (
            self.use_taichi
            and self.memory_rank is None
            and (not torch.is_grad_enabled() or self._taichi_dispatchable())
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
        r"""extra_repr() -> str

        String representation of the layer's key parameters.

        Returns:
            str: Comma-separated ``d_model``, ``k_stats``, and optional
            ``memory_rank``.
        """
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
