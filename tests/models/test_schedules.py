"""Behavioral tests for learning-rate schedulers."""

from __future__ import annotations

import pytest
import torch

from qwendopamine.training.schedules import (
    LinearWarmupScheduler,
    build_scheduler,
)


def test_when_build_scheduler_cosine_then_returns_linear_warmup_scheduler() -> None:
    param = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = build_scheduler(optimizer, name="cosine", warmup_steps=100, min_lr=1e-5)

    assert isinstance(scheduler, LinearWarmupScheduler)


def test_when_build_scheduler_unknown_then_raises_key_error() -> None:
    param = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([param], lr=1e-3)

    with pytest.raises(KeyError, match="Unknown scheduler: unknown_name"):
        build_scheduler(optimizer, name="unknown_name")


def test_when_linear_warmup_scheduler_steps_then_scales_learning_rate() -> None:
    param = torch.nn.Parameter(torch.randn(2, 2))
    initial_lr = 1e-3
    optimizer = torch.optim.AdamW([{"params": [param], "initial_lr": initial_lr}], lr=initial_lr)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    warmup_steps = 10
    scheduler = LinearWarmupScheduler(optimizer, base_scheduler, warmup_steps=warmup_steps, min_lr=1e-5)

    # Step through warmup phase
    for i in range(1, warmup_steps + 1):
        optimizer.step()
        scheduler.step()
        expected_lr = initial_lr * (i / warmup_steps)
        current_lr = optimizer.param_groups[0]["lr"]
        assert pytest.approx(current_lr) == expected_lr

    # Base scheduler takes over after warmup
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < initial_lr


def test_when_linear_warmup_scheduler_state_dict_then_serializes_and_restores() -> None:
    param = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([{"params": [param], "initial_lr": 1e-3}], lr=1e-3)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    scheduler = LinearWarmupScheduler(optimizer, base_scheduler, warmup_steps=10, min_lr=1e-5)

    state = scheduler.state_dict()
    assert isinstance(state, dict)

    scheduler.load_state_dict(state)
