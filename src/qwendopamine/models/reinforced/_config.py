# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""Configuration dataclass for the Gated Reward Network."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

__all__ = ["GatedRewardNetConfig"]


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
