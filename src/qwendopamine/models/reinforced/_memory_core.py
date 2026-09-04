# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

"""Core Gated Delta Rule 2 memory update with coupled write/erase gates."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["DeltaMemoryCore"]


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
        # Apply channel-wise decay: U' = (1 - omega_E) * U
        U_new = (1.0 - omega_E) * U
        # Absorb the rank-1 (omega_W ⊙ e) k^T update into V: V_new = V + (omega_W ⊙ e) ⊗ k.
        w_e = omega_W.squeeze(-1) * e_t  # (B, d)
        v_col = k_t.unsqueeze(-1)  # (B, d, 1)
        V_new = V + v_col * w_e.unsqueeze(-1)
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
        r"""extra_repr() -> str

        Return a string with the extra representation of the module."""
        rank = f", memory_rank={self.memory_rank}" if self.memory_rank else ""
        return f"d_model={self.d_model}, use_short_conv={self.use_short_conv}{rank}"
