"""Port for parallel reward branch introspection.

This protocol exists to make the interaction boundary between model internals
and external tooling explicit. Training and evaluation utilities only need to
inspect the gate projection; they should not depend on the exact module
attribute layout.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class ParallelRewardPort(Protocol):
    """Minimal contract for parallel reward branch introspection."""

    @property
    def reward_gate_proj(self) -> nn.Module:
        """Data-dependent gate projection for the parallel reward branch."""
        ...
