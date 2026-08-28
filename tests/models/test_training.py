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
from qwendopamine.training.schedules import LinearWarmupScheduler


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
    tracker.update("loss", 1.75)
    tracker.update("ppl", 3.49)

    assert tracker.values["loss"] == 1.75
    assert tracker.get_mean("loss") == pytest.approx(1.50, rel=1e-6)
    assert tracker.get_history("loss") == [1.25, 1.75]

    state = tracker.state_dict()
    assert state == {"loss": 1.75, "ppl": 3.49}
    state["loss"] = 9.99
    assert tracker.values["loss"] == 1.75

    tracker.reset()
    assert len(tracker.values) == 0
    assert tracker.get_mean("loss") == 0.0


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

    config = TrainConfig(
        max_steps=4, grad_accum_steps=2, max_grad_norm=1.0, mixed_precision="fp16"
    )
    loop = TrainingLoop(model, optimizer, scheduler, config)

    # Snapshot parameters before training.
    params_before = [p.clone().detach() for p in model.parameters()]

    batches = [{"x": torch.randn(2, 8), "labels": torch.randn(2, 8)} for _ in range(10)]

    loop.run(batches)

    assert loop.global_step == 4

    # Verify at least one parameter actually changed (optimizer took steps).
    params_after = [p.detach() for p in model.parameters()]
    any_changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert any_changed, "TrainingLoop ran but no parameters were updated"
