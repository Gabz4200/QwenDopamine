"""M9: parallel reward monitoring must debug-log, not silently swallow."""

from __future__ import annotations

import logging

import pytest


def test_parallel_reward_except_logs_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When cache reading fails, the helper must log at DEBUG, not pass.

    Review M9: the previous code did ``except (AttributeError, IndexError):
    pass`` which made misconfigured models invisible. The fix logs at
    DEBUG so a user can diagnose without crashing the training loop.
    """
    from torch import nn

    from qwendopamine.training import parallel_reward as pr

    class _Layer(nn.Module):
        layer_idx = 0
        reward_branch = nn.Linear(1, 1)
        reward_gate_proj = nn.Linear(1, 1)

    class _Model(nn.Module):
        layers = nn.ModuleList([_Layer()])

        def __init__(self) -> None:
            super().__init__()

    class _BadCache:
        # Force the except branch by having ``layers`` be an empty list
        # so ``layers[layer_idx]`` raises IndexError.
        layers = []

    model = _Model()
    with caplog.at_level(logging.DEBUG, logger=pr.__name__):
        metrics = pr.collect_parallel_reward_metrics(model, past_key_values=_BadCache())
    assert "parallel_reward/active_layers" in metrics
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("parallel_reward" in r.message for r in debug_records), (
        f"Expected a DEBUG log; got {[r.message for r in caplog.records]!r}"
    )
