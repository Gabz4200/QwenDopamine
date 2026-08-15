"""Behavioral tests for training components (loop, schedules, freezing, metrics)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwendopamine.training.freezing import (
    freeze_module,
    set_trainable,
    trainable_parameters,
    unfreeze_module,
)
from qwendopamine.training.loop import TrainConfig, TrainingLoop
from qwendopamine.training.metrics import MetricTracker
from qwendopamine.training.schedules import (
    LinearWarmupScheduler,
    build_scheduler,
)


def test_when_freeze_and_unfreeze_module_then_toggles_parameter_grad() -> None:
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 2))

    freeze_module(model)
    assert len(trainable_parameters(model)) == 0
    assert all(not p.requires_grad for p in model.parameters())

    unfreeze_module(model)
    assert len(trainable_parameters(model)) == len(list(model.parameters()))
    assert all(p.requires_grad for p in model.parameters())

    set_trainable(model[0], False)
    assert all(not p.requires_grad for p in model[0].parameters())
    assert all(p.requires_grad for p in model[1].parameters())


def test_when_metric_tracker_updated_then_records_and_exports_dict() -> None:
    tracker = MetricTracker()
    tracker.update("loss", 1.25)
    tracker.update("ppl", 3.49)

    state = tracker.state_dict()
    assert state == {"loss": 1.25, "ppl": 3.49}
    state["loss"] = 9.99
    assert tracker.values["loss"] == 1.25


def test_when_build_scheduler_cosine_then_returns_linear_warmup_scheduler() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = build_scheduler(optimizer, name="cosine", warmup_steps=100, min_lr=1e-5)

    assert isinstance(scheduler, LinearWarmupScheduler)


def test_when_build_scheduler_unknown_then_raises_key_error() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([param], lr=1e-3)

    with pytest.raises(KeyError, match="Unknown scheduler: unknown_name"):
        build_scheduler(optimizer, name="unknown_name")


def test_when_linear_warmup_scheduler_steps_then_scales_learning_rate() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    initial_lr = 1e-3
    optimizer = torch.optim.AdamW([{"params": [param], "initial_lr": initial_lr}], lr=initial_lr)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    warmup_steps = 10
    scheduler = LinearWarmupScheduler(optimizer, base_scheduler, warmup_steps=warmup_steps, min_lr=1e-5)

    for i in range(1, warmup_steps + 1):
        optimizer.step()
        scheduler.step()
        expected_lr = initial_lr * (i / warmup_steps)
        current_lr = optimizer.param_groups[0]["lr"]
        assert pytest.approx(current_lr) == expected_lr

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < initial_lr


def test_when_linear_warmup_scheduler_state_dict_then_serializes_and_restores() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([{"params": [param], "initial_lr": 1e-3}], lr=1e-3)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    scheduler = LinearWarmupScheduler(optimizer, base_scheduler, warmup_steps=10, min_lr=1e-5)

    state = scheduler.state_dict()
    assert isinstance(state, dict)
    scheduler.load_state_dict(state)


class ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        out = self.fc(x)
        if labels is not None:
            return {"loss": nn.functional.mse_loss(out, labels)}
        return {"logits": out}


def test_when_training_loop_run_then_advances_global_steps_and_optimizes() -> None:
    model = ToyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    base_sched = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    scheduler = LinearWarmupScheduler(
        optimizer, base_sched, warmup_steps=2, min_lr=1e-5
    )

    config = TrainConfig(max_steps=4, grad_accum_steps=2, max_grad_norm=1.0, mixed_precision="fp16")
    loop = TrainingLoop(model, optimizer, scheduler, config)

    batches = [
        {"x": torch.randn(2, 8), "labels": torch.randn(2, 8)}
        for _ in range(10)
    ]

    loop.run(batches)

    assert loop.global_step == 4
