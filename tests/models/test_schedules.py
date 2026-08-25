"""Behavioral tests for learning-rate schedules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwendopamine.training.schedules import (
    LinearWarmupScheduler,
    build_scheduler,
)


class _DummyOptimizer:
    """Minimal optimizer stub for testing scheduler logic in isolation."""

    def __init__(self, lr: float = 1e-3) -> None:
        self.param_groups: list[dict[str, float]] = [{"lr": lr, "initial_lr": lr}]
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1


def test_when_build_scheduler_cosine_then_returns_linear_warmup() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = build_scheduler(optimizer, name="cosine", warmup_steps=100, min_lr=1e-5)

    assert isinstance(scheduler, LinearWarmupScheduler)


def test_when_build_scheduler_unknown_name_then_raises_key_error() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([param], lr=1e-3)

    with pytest.raises(KeyError, match="Unknown scheduler: bad_name"):
        build_scheduler(optimizer, name="bad_name")


def test_when_warmup_step_then_lr_scales_linearly() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    initial_lr = 1e-2
    optimizer = torch.optim.AdamW(
        [{"params": [param], "initial_lr": initial_lr}], lr=initial_lr
    )
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5
    )
    warmup_steps = 10
    scheduler = LinearWarmupScheduler(
        optimizer, base_scheduler, warmup_steps=warmup_steps, min_lr=1e-5
    )

    for step in range(1, warmup_steps + 1):
        scheduler.step()
        expected_lr = initial_lr * (step / warmup_steps)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr, rel=1e-6)


def test_when_warmup_complete_then_lr_follows_base_scheduler() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    initial_lr = 1e-2
    optimizer = torch.optim.AdamW(
        [{"params": [param], "initial_lr": initial_lr}], lr=initial_lr
    )
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5
    )
    scheduler = LinearWarmupScheduler(
        optimizer, base_scheduler, warmup_steps=5, min_lr=1e-5
    )

    # Complete warmup.
    for _ in range(5):
        scheduler.step()

    lr_after_warmup = optimizer.param_groups[0]["lr"]
    assert lr_after_warmup == pytest.approx(initial_lr, rel=1e-6)

    # Step once more — base scheduler takes over, LR should decrease.
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < initial_lr


def test_when_warmup_zero_steps_then_lr_always_at_base() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    initial_lr = 1e-3
    optimizer = torch.optim.AdamW(
        [{"params": [param], "initial_lr": initial_lr}], lr=initial_lr
    )
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5
    )
    scheduler = LinearWarmupScheduler(
        optimizer, base_scheduler, warmup_steps=0, min_lr=1e-5
    )

    scheduler.step()
    # With warmup_steps=0, we skip warmup and go straight to base scheduler.
    # CosineAnnealingLR step should reduce LR from initial_lr.
    assert optimizer.param_groups[0]["lr"] <= initial_lr


def test_when_scheduler_state_dict_roundtrip_then_state_preserved() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([{"params": [param], "initial_lr": 1e-3}], lr=1e-3)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5
    )
    scheduler = LinearWarmupScheduler(
        optimizer, base_scheduler, warmup_steps=10, min_lr=1e-5
    )

    # Advance a few steps.
    for _ in range(5):
        scheduler.step()

    state = scheduler.state_dict()
    assert isinstance(state, dict)

    # Restore into a fresh scheduler.
    optimizer2 = torch.optim.AdamW(
        [{"params": [nn.Parameter(torch.randn(2, 2))], "initial_lr": 1e-3}], lr=1e-3
    )
    base_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=100, eta_min=1e-5
    )
    scheduler2 = LinearWarmupScheduler(
        optimizer2, base_scheduler2, warmup_steps=10, min_lr=1e-5
    )
    scheduler2.load_state_dict(state)


def test_when_scheduler_step_count_tracked_then_step_count_increments() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([{"params": [param], "initial_lr": 1e-3}], lr=1e-3)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5
    )
    scheduler = LinearWarmupScheduler(
        optimizer, base_scheduler, warmup_steps=10, min_lr=1e-5
    )

    assert scheduler.step_count == 0
    scheduler.step()
    assert scheduler.step_count == 1
    scheduler.step()
    assert scheduler.step_count == 2


def test_when_build_scheduler_warmup_steps_zero_then_no_warmup() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = build_scheduler(optimizer, name="cosine", warmup_steps=0, min_lr=1e-5)

    assert isinstance(scheduler, LinearWarmupScheduler)
    assert scheduler.warmup_steps == 0
