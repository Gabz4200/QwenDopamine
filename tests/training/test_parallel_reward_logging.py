"""M9: parallel reward must fail fast on cache mismatch, not silently swallow."""

from __future__ import annotations

import pytest
from torch import nn


def test_parallel_reward_raises_on_layer_count_mismatch() -> None:
    """When ``past_key_values.layers`` has fewer entries than the model's
    layer_idx, ``collect_parallel_reward_metrics`` must raise IndexError.

    The previous code caught this with ``except (AttributeError, IndexError):
    pass``, hiding misconfigured caches. Now it fails fast.
    """
    from qwendopamine.training import parallel_reward as pr

    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_idx = 0
            self.reward_branch = nn.Module()
            self.reward_branch.reward_gate_proj = nn.Linear(1, 1)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Layer()])

    class _BadCache:
        layers: list

        def __init__(self) -> None:
            self.layers = []

    model = _Model()
    with pytest.raises(IndexError, match="layer_idx=0"):
        pr.collect_parallel_reward_metrics(model, past_key_values=_BadCache())
