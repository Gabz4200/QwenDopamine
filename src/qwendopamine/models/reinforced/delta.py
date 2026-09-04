# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

r"""RL-augmented Gated Delta Rule 2 components: ValueBaselineEMA, AdvantageGate, DeltaMemoryCore, and ReinforcedDeltaLayer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._advantage_gate import AdvantageGate
from ._config import GatedRewardNetConfig
from ._layer import ReinforcedDeltaLayer, normalize_reward_for_advantage
from ._memory_core import DeltaMemoryCore
from ._value_baseline import ValueBaselineEMA

__all__ = [
    "AdvantageGate",
    "DeltaMemoryCore",
    "GatedRewardNetConfig",
    "ReinforcedDeltaLayer",
    "ValueBaselineEMA",
]


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

        # Centralised cache-field-name table. The reward-namespaced
        # attributes (``reward_recurrent_state`` etc.) take priority;
        # the un-prefixed names are accepted as fallbacks so legacy
        # caches built before the namespace was introduced keep
        # working. Touching this list is the only place the cache
        # schema lives — the write side in ``_write_cache`` mirrors it.
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
                if baseline is None:
                    baseline = getattr(lc, "value_baseline", None)
                running_mean = getattr(lc, "reward_running_mean", None)
                if running_mean is None:
                    running_mean = getattr(lc, "running_mean", None)
                running_std = getattr(lc, "reward_running_std", None)
                if running_std is None:
                    running_std = getattr(lc, "running_std", None)
                return rec, conv_pair, baseline, running_mean, running_std
            return None, None, None, None, None
        if isinstance(past_key_values, dict):
            rec = past_key_values.get("recurrent_state")
            if rec is None:
                rec = past_key_values.get("reward_recurrent_state")
            if rec is None:
                rec = past_key_values.get("recurrent_states")
                if isinstance(rec, dict):
                    rec = next((v for v in rec.values() if v is not None), None)
                elif isinstance(rec, (list, tuple)) and rec:
                    rec = rec[0]
            conv = past_key_values.get("conv_state")
            if conv is None:
                conv = past_key_values.get("reward_conv_states")
            if conv is None:
                conv = past_key_values.get("conv_states")
            conv_pair = _extract_conv(conv)
            baseline = past_key_values.get("value_baseline")
            if baseline is None:
                baseline = past_key_values.get("reward_value_baseline")
            running_mean = past_key_values.get("running_mean")
            if running_mean is None:
                running_mean = past_key_values.get("reward_running_mean")
            running_std = past_key_values.get("running_std")
            if running_std is None:
                running_std = past_key_values.get("reward_running_std")
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
        """Broadcast ``reward_values`` to ``(B, L, k_stats)``.

        Accepts 1D, 2D, or 3D inputs. The 1D path is heuristic (a vector
        that matches the sequence length is treated as a per-token
        scalar; otherwise it is treated as a per-channel vector). If
        the shape is ambiguous the caller should pass a 2D or 3D
        tensor explicitly.

        Empty / scalar inputs default to zero reward.
        """
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
            # Otherwise: must be a per-channel vector of size k_stats.
            # If neither size matches, raise: the heuristic is unsafe.
            if reward_values.size(0) == batch_size:
                # (B,) -> per-sample scalar broadcast across (L, 1).
                return reward_values.view(batch_size, 1, 1).expand(-1, seq_len, 1)
            if reward_values.size(0) != self.k_stats:
                raise ValueError(
                    f"1D reward_values of size {reward_values.size(0)} does not match "
                    f"seq_len={seq_len}, batch_size={batch_size}, or k_stats={self.k_stats}. "
                    "Pass a 2D or 3D tensor to disambiguate."
                )
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
        r"""GatedRewardNet.forward(hidden_states, reward_values=None, past_key_values=None, output_attentions=False, use_cache=True, **kwargs) -> tuple[Tensor, None, Any]

        Apply the gated reward network recurrence.

        Args:
            hidden_states (Tensor): Input ``[B, T, D]`` or ``[B, D]``.
            reward_values (Tensor | None): Reward signal ``[B, T]`` or similar.
            past_key_values (Any): Optional cache for incremental decoding.
            output_attentions (bool): Whether to return attention weights.
            use_cache (bool): Persist recurrent state for next call.
            **kwargs: Extra arguments ignored by this layer.

        Returns:
            tuple[Tensor, None, Any]: ``(output, None, new_cache)`` where
            ``new_cache`` holds recurrent state and running norm stats.
        """
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
        r"""extra_repr() -> str

        Return a string with the extra representation of the module."""
        rank = f", memory_rank={self.memory_rank}" if self.memory_rank else ""
        return f"hidden_size={self.hidden_size}, k_stats={self.k_stats}{rank}"
