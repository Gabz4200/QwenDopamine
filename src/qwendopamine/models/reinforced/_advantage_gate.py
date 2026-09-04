# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

"""Plasticity-aware advantage gate for the reinforced delta layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["AdvantageGate"]


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
            self.advantage_proj.bias.zero_()
            self.advantage_proj.weight.zero_()
            self.plasticity_proj.bias.zero_()
            self.plasticity_proj.weight.zero_()
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
        r"""extra_repr() -> str

        Return a string with the extra representation of the module."""
        return (
            f"k_stats={self.k_stats}, dropout={self.dropout}, "
            f"legacy_coupled={self.legacy_coupled}"
        )
