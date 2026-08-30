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
    r"""PPO-inspired Advantage Gate producing global modulation scalar ω_t ∈ (0, 2).

    Collapses the vectorial advantage A_t ∈ ℝ^k into a single scalar gate via a
    learned linear projection followed by 2·sigmoid, emulating PPO's clipping
    behavior with smooth gradients:

        ω_t = 2 · σ(W_a A_t + b_a)

    This restricts ω_t ∈ (0, 2), where:
    - ω_t ≈ 0 (A_t ≪ 0): Freeze memory (both write and erase close).
    - ω_t ≈ 1 (A_t ≈ 0): Standard Delta Rule operation.
    - ω_t ≈ 2 (A_t ≫ 0): Boost write/erase for strong positive advantage.

    Args:
        k_stats (int): Dimension of advantage vector A_t.
        dropout (float, optional): Dropout probability on advantage features.
            Default: 0.0.

    Shape:
        - A_t: (B, k_stats)
        - Returns: ω_t (B, 1)
    """

    def __init__(self, k_stats: int, dropout: float = 0.0) -> None:
        super().__init__()

        if k_stats <= 0:
            raise ValueError("k_stats must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.k_stats = k_stats
        self.dropout = dropout
        self.advantage_proj = nn.Linear(k_stats, 1)

        with torch.no_grad():
            self.advantage_proj.bias.zero_()
            self.advantage_proj.weight.zero_()

    def forward(self, A_t: Tensor) -> Tensor:
        """
        Args:
            A_t: (B, k_stats) - Advantage vector.

        Returns:
            ω_t: (B, 1) - Global modulation scalar in (0, 2).
        """
        if A_t.dim() != 2:
            raise ValueError(f"Expected A_t shape (B, k_stats), got {A_t.shape}.")
        if A_t.size(-1) != self.k_stats:
            raise ValueError(
                f"Expected k_stats={self.k_stats}, got {A_t.size(-1)}."
            )

        if self.training and self.dropout > 0.0:
            A_t = F.dropout(A_t, p=self.dropout, training=True)

        # ω_t = 2 · σ(W_a A_t + b_a)
        omega_t = 2.0 * torch.sigmoid(self.advantage_proj(A_t))  # (B, 1)
        return omega_t

    def extra_repr(self) -> str:
        return f"k_stats={self.k_stats}, dropout={self.dropout}"


class DeltaMemoryCore(nn.Module):
    r"""Core Gated Delta Rule 2 memory update with coupled write/erase gates.

    Implements the vector-matrix update:
        S_{t+1} = (1 - ω_t E_t) ⊙ S_t + (ω_t W_t) ⊙ (e_t k_t^T)

    where:
    - S_t ∈ ℝ^{d×d} is the fast weight matrix (state).
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

    Shape:
        - x: (B, d_model)
        - ω_t: (B, 1)
        - S_prev: (B, d_model, d_model)
        - Returns: S_next (B, d_model, d_model)
    """

    def __init__(
        self,
        d_model: int,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if conv_size <= 0:
            raise ValueError("conv_size must be positive.")

        self.d_model = d_model
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size

        # Projections for key, value
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Short convolutions for causal smoothing
        if use_short_conv:
            from qwendopamine.models.gdn2.gdn2 import ShortConvolution
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

    def forward(
        self,
        x: Tensor,
        omega_t: Tensor,
        S_prev: Tensor,
        k_cache: Tensor | None = None,
        v_cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Args:
            x: (B, d_model) - Input features.
            omega_t: (B, 1) - Global advantage modulation.
            S_prev: (B, d, d) - Previous state matrix.
            k_cache: Optional cached conv state for k.
            v_cache: Optional cached conv state for v.

        Returns:
            S_next: (B, d, d) - Updated state.
            k_cache_out: Updated k conv cache.
            v_cache_out: Updated v conv cache.
        """
        _, _d = x.shape

        # Project and optionally apply short conv
        k_t = self.k_proj(x)  # (B, d)
        v_t = self.v_proj(x)  # (B, d)

        k_cache_out = None
        v_cache_out = None

        if self.use_short_conv:
            k_t, k_cache_out = self.k_conv1d(k_t.unsqueeze(1), k_cache)
            k_t = k_t.squeeze(1)
            v_t, v_cache_out = self.v_conv1d(v_t.unsqueeze(1), v_cache)
            v_t = v_t.squeeze(1)

        # Write and Erase gates (channel-wise, sigmoid activated)
        W_t = torch.sigmoid(self.w_proj(x))  # (B, d)
        E_t = torch.sigmoid(self.e_proj(x))  # (B, d)

        # Residual error: e_t = v_t - S_t k_t
        # S_prev: (B, d, d), k_t: (B, d) -> pred: (B, d)
        pred = torch.bmm(S_prev, k_t.unsqueeze(-1)).squeeze(-1)  # (B, d)
        e_t = v_t - pred  # (B, d)

        # Outer product: e_t k_t^T -> (B, d, d)
        outer_prod = torch.bmm(e_t.unsqueeze(-1), k_t.unsqueeze(1))  # (B, d, d)

        # Coupled modulation: ω_t scales both write and erase
        # omega_t: (B, 1) -> broadcast to (B, d, 1) for gates
        # Clamp to keep decay in [0,1] (no sign-flip) and write in [0,2]
        omega_W = (omega_t.unsqueeze(-1) * W_t.unsqueeze(-1)).clamp(max=2.0)  # (B, d, 1) in [0,2]
        omega_E = (omega_t.unsqueeze(-1) * E_t.unsqueeze(-1)).clamp(max=1.0)  # (B, d, 1) in [0,1]

        decay_term = 1.0 - omega_E  # (B, d, 1) in [0,1]

        # S_{t+1} = (1 - ω E) ⊙ S + (ω W) ⊙ (e k^T)
        S_next = (decay_term * S_prev) + (omega_W * outer_prod)

        return S_next, k_cache_out, v_cache_out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, use_short_conv={self.use_short_conv}"


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

    Shape (per step):
        - x: (B, d_model)
        - reward_values: (B, k_raw) or broadcastable - raw rewards.
        - S_prev: (B, d, d) or None for initialization.
        - V_prev: (B, k_stats) or None for initialization.
        - Returns: o_t (B, d_model), S_next (B, d, d), V_t (B, k_stats)
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

        self.d_model = d_model
        self.k_stats = k_stats
        self.reward_dropout = reward_dropout
        self.advantage_dropout = advantage_dropout

        # Statistics extraction and normalization (local import to avoid circular)
        from qwendopamine.models.blocks.reward import (
            LearnableSoftsign,
            RewardStatisticsExtractor,
        )
        self.stats_extractor = RewardStatisticsExtractor(reward_dropout=reward_dropout)
        self.stats_normalizer = LearnableSoftsign(per_channel=True, num_channels=k_stats)

        # RL components
        self.baseline_tracker = ValueBaselineEMA(d_model, k_stats, init_alpha=init_alpha)
        self.advantage_gate = AdvantageGate(k_stats, dropout=advantage_dropout)

        # Core memory
        self.memory_core = DeltaMemoryCore(
            d_model=d_model,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
        )

        # Query projection and FiLM conditioning
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.reward_encoder = reward_encoder  # Returns (gamma, beta) from R_stats

        # Small-gain init for stability (keeps FiLM near identity at start)
        torch.nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)

    def forward(
        self,
        x: Tensor,
        reward_values: Tensor,
        S_prev: Tensor | None = None,
        V_prev: Tensor | None = None,
        k_cache: Tensor | None = None,
        v_cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """
        Args:
            x: (B, d_model) - Input features for current token.
            reward_values: Raw reward tensor. Shape flexible:
                (B, k_raw), (k_raw,), (B,), scalar, etc.
                Will be normalized by stats_extractor to (B, k_stats).
            S_prev: (B, d, d) - Previous fast weight matrix.
            V_prev: (B, k_stats) - Previous EMA baseline.
            k_cache: Optional cached conv state for k.
            v_cache: Optional cached conv state for v.

        Returns:
            o_t: (B, d_model) - Output features (readout).
            S_next: (B, d, d) - Updated fast weight matrix.
            V_t: (B, k_stats) - Updated EMA baseline.
            k_cache_out: Updated k conv cache.
            v_cache_out: Updated v conv cache.
        """
        B = x.size(0)

        if S_prev is None:
            S_prev = torch.zeros(B, self.d_model, self.d_model, device=x.device, dtype=x.dtype)
        if V_prev is None:
            V_prev = torch.zeros(B, self.k_stats, device=x.device, dtype=x.dtype)

        # 1. Reward Statistics Extraction & Normalization
        R_stats = self.stats_extractor(reward_values, batch_size=B, seq_len=1)  # (B, 1, k_stats)
        R_stats = self.stats_normalizer(R_stats).squeeze(1)  # (B, k_stats)

        # 2. RL: Baseline Tracking & Advantage Gate
        V_t, A_t = self.baseline_tracker(x, R_stats, V_prev)  # (B, k_stats) each
        omega_t = self.advantage_gate(A_t)  # (B, 1)

        # 3. FiLM Conditioning on Query (using normalized R_stats)
        q_t = self.q_proj(x)  # (B, d)
        gamma_t, beta_t = self.reward_encoder(R_stats)  # (B, d) each
        q_prime_t = gamma_t * q_t + beta_t  # (B, d)

        # 4. Memory Update (DeltaMemoryCore)
        S_next, k_cache_out, v_cache_out = self.memory_core(
            x, omega_t, S_prev, k_cache=k_cache, v_cache=v_cache
        )

        # 5. Readout with FiLM-modulated Query
        q_prime_unsig = q_prime_t.unsqueeze(-1)  # (B, d, 1)
        o_t = torch.bmm(S_next, q_prime_unsig).squeeze(-1)  # (B, d)

        return o_t, S_next, V_t, k_cache_out, v_cache_out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, k_stats={self.k_stats}"
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
        )
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        torch.nn.init.xavier_uniform_(self.output_proj.weight, gain=2**-2.5)
    def _get_cache(self, past_key_values: Any) -> tuple[Tensor | None, tuple[Tensor | None, Tensor | None] | None]:
        if past_key_values is None:
            return None, None
        if hasattr(past_key_values, "layers"):
            layers = getattr(past_key_values, "layers", [])
            if self.layer_idx is not None and self.layer_idx < len(layers):
                lc = layers[self.layer_idx]
                rec = getattr(lc, "recurrent_state", None)
                if rec is None:
                    rec = getattr(lc, "recurrent_states", None)
                    if isinstance(rec, (list, tuple)) and rec:
                        rec = rec[0]
                conv = getattr(lc, "conv_state", None)
                if conv is None:
                    conv = getattr(lc, "conv_states", None)
                if isinstance(conv, (list, tuple)) and len(conv) >= 2:
                    conv = (conv[0], conv[1])
                return rec, conv
            return None, None
        if isinstance(past_key_values, dict):
            rec = past_key_values.get("recurrent_state")
            conv = past_key_values.get("conv_state")
            if isinstance(conv, (list, tuple)) and len(conv) >= 2:
                conv = (conv[0], conv[1])
            return rec, conv
        return None, None

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

    def forward(self, hidden_states: Tensor, reward_values: Tensor | None = None, past_key_values: Any = None, output_attentions: bool = False, use_cache: bool = True, **kwargs: Any) -> tuple[Tensor, None, Any]:
        if hidden_states.dim() == 3:
            B, L, _ = hidden_states.shape
            seq_len = L
        elif hidden_states.dim() == 2:
            B, _ = hidden_states.shape
            hidden_states = hidden_states.unsqueeze(1)
            seq_len = 1
        else:
            raise ValueError(f"Expected hidden_states dim 2 or 3, got {hidden_states.dim()}.")
        rec_state, conv_state = self._get_cache(past_key_values)
        S_curr = rec_state
        V_curr = None
        k_cache = conv_state[0] if conv_state is not None else None
        v_cache = conv_state[1] if conv_state is not None else None
        reward_values = self._normalize_reward_tensor(
            reward_values,
            batch_size=B,
            seq_len=seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
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
            new_cache = {"recurrent_state": S_curr, "conv_state": (k_cache, v_cache)}
        return out, None, new_cache
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, k_stats={self.k_stats}"
