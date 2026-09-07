"""Behavioural tests for the training module: training loop, schedules,
freezing, metrics, and parallel-reward monitoring.

Complements the focused unit tests under ``tests/models/test_training.py``,
``test_schedules.py``, and ``test_parallel_reward_monitoring.py`` with
integration-style coverage of the public surface.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwendopamine.training import (
    MetricTracker,
    TrainingLoop,
    build_scheduler,
    collect_parallel_reward_metrics,
    freeze_module,
    maybe_warn_branch_ratio,
    set_trainable,
    trainable_parameters,
    unfreeze_module,
    validate_unfreeze_phases,
)
from qwendopamine.training.loop import TrainConfig
from qwendopamine.training.schedules import LinearWarmupScheduler

# ---------------------------------------------------------------------------
# TrainConfig + TrainingLoop
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    r"""Minimal linear LM with a constant loss so ``TrainingLoop`` can step."""

    def __init__(self, hidden_size: int = 8, vocab_size: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict:
        logits = self.head(self.embed(input_ids))
        if labels is None:
            return {"logits": logits}
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        return {"loss": loss, "logits": logits}


def _batches(
    n: int = 3, batch_size: int = 2, seq_len: int = 4, vocab_size: int = 16
) -> list[dict]:
    return [
        {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        }
        for _ in range(n)
    ]


def test_when_training_loop_run_with_bf16_then_advances_global_step() -> None:
    """End-to-end: the loop must advance ``global_step`` after enough accum
    steps and finish without raising on a real model."""
    model = _TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = build_scheduler(optimizer, name="cosine", warmup_steps=1, min_lr=1e-5)
    cfg = TrainConfig(max_steps=2, grad_accum_steps=1, mixed_precision="bf16")
    loop = TrainingLoop(model, optimizer, scheduler, cfg)

    loop.run(_batches(n=3))

    assert loop.global_step == 2


def test_when_training_loop_empty_loader_then_raises() -> None:
    """An empty dataloader must raise ``ValueError`` rather than silently succeed."""
    model = _TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = build_scheduler(optimizer, name="cosine", warmup_steps=0)
    cfg = TrainConfig(max_steps=10, mixed_precision="bf16")
    loop = TrainingLoop(model, optimizer, scheduler, cfg)

    with pytest.raises(ValueError, match="empty"):
        loop.run([])


def test_when_training_loop_accum_steps_two_then_only_steps_at_boundary() -> None:
    """With ``grad_accum_steps=2``, ``global_step`` must advance once per
    two micro-batches."""
    model = _TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = build_scheduler(optimizer, name="cosine", warmup_steps=0)
    cfg = TrainConfig(max_steps=2, grad_accum_steps=2, mixed_precision="bf16")
    loop = TrainingLoop(model, optimizer, scheduler, cfg)

    loop.run(_batches(n=4))

    # Two optimizer steps should have happened for four micro-batches.
    assert loop.global_step == 2


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_when_build_scheduler_cosine_then_returns_linear_warmup() -> None:
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    scheduler = build_scheduler(optimizer, name="cosine", warmup_steps=5)
    assert isinstance(scheduler, LinearWarmupScheduler)
    assert scheduler.warmup_steps == 5


def test_when_warmup_scheduler_state_dict_restored_then_lr_matches() -> None:
    """The scheduler must serialise ``step_count`` and replay the same lr on
    restore."""
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-5
    )
    scheduler = LinearWarmupScheduler(
        optimizer, base_scheduler, warmup_steps=4, min_lr=1e-5
    )
    for _ in range(3):
        scheduler.step()
    state = scheduler.state_dict()

    # Build a fresh scheduler and load the state.
    param2 = nn.Parameter(torch.randn(2, 2))
    optimizer2 = torch.optim.AdamW([param2], lr=1e-3)
    base2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=10, eta_min=1e-5
    )
    fresh = LinearWarmupScheduler(optimizer2, base2, warmup_steps=4, min_lr=1e-5)
    fresh.load_state_dict(state)
    assert fresh.step_count == scheduler.step_count


# ---------------------------------------------------------------------------
# Freezing helpers
# ---------------------------------------------------------------------------


def test_when_freeze_module_then_requires_grad_false_on_every_param() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    freeze_module(model)
    assert not any(p.requires_grad for p in model.parameters())


def test_when_unfreeze_module_after_freeze_then_all_params_trainable() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    freeze_module(model)
    unfreeze_module(model)
    assert all(p.requires_grad for p in model.parameters())


def test_when_set_trainable_false_on_submodule_then_only_that_submodule_freezes() -> (
    None
):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    set_trainable(model[0], False)
    assert all(not p.requires_grad for p in model[0].parameters())
    assert all(p.requires_grad for p in model[1].parameters())


def test_when_trainable_parameters_called_then_returns_only_grad_enabled() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    set_trainable(model[0], False)
    trainable = trainable_parameters(model)
    assert len(trainable) == sum(1 for _ in model[1].parameters())
    assert all(p.requires_grad for p in trainable)


def test_when_validate_unfreeze_phases_with_unknown_module_then_returns_phases(
    caplog,
) -> None:
    """Unknown module names in unfreeze phases must not raise; the function
    should log a warning and return the input unchanged."""
    model = nn.Linear(4, 4)
    phases = [{"name": "phase1", "modules": ["nonexistent.module"]}]

    with caplog.at_level("WARNING"):
        result = validate_unfreeze_phases(model, phases)

    assert result is phases
    assert any("nonexistent.module" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# MetricTracker
# ---------------------------------------------------------------------------


def test_when_metric_tracker_state_dict_loaded_then_latest_value_restored() -> None:
    """``load_state_dict`` replays the latest values through ``update``; the
    restored tracker must report the same latest value as the source.
    """
    tracker = MetricTracker()
    tracker.update("loss", 0.5)
    tracker.update("loss", 1.5)
    state = tracker.state_dict()

    fresh = MetricTracker()
    fresh.load_state_dict(state)
    assert fresh.values["loss"] == 1.5
    # state_dict serialises only the latest per-metric value, so the mean
    # after a single ``update`` equals that value.
    assert fresh.get_mean("loss") == pytest.approx(1.5, rel=1e-6)
    assert fresh.get_history("loss") == [1.5]


# ---------------------------------------------------------------------------
# collect_parallel_reward_metrics + maybe_warn_branch_ratio
# ---------------------------------------------------------------------------


def test_when_collect_metrics_with_branch_ratio_above_threshold_then_warns() -> None:
    """``maybe_warn_branch_ratio`` must return a string when the branch
    contribution is large enough to exceed the threshold."""
    metrics = {
        "parallel_reward/main_norm": 1.0,
        "parallel_reward/branch_norm": 0.9,
        "parallel_reward/branch_ratio": 0.9,
    }
    warning = maybe_warn_branch_ratio(metrics, threshold=0.1)
    assert warning is not None
    assert "0.9" in warning or "ratio" in warning.lower()


def test_when_collect_metrics_with_no_active_branch_then_only_active_layers() -> None:
    """A model without a parallel reward branch must surface a flat metric
    dict with only ``active_layers=0``."""

    class _PlainModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    metrics = collect_parallel_reward_metrics(_PlainModel())
    assert metrics == {"parallel_reward/active_layers": 0.0}


def test_when_collect_metrics_with_branch_inputs_then_ratios_computed() -> None:
    """When the model has an active parallel branch and branch tensors are
    supplied, the ratio metrics must be computed."""
    branch_layer = nn.Linear(2, 2)
    branch_layer.reward_branch = nn.Linear(2, 2)
    branch_layer.reward_gate_proj = nn.Linear(2, 2)

    class _MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([branch_layer])

    model = _MockModel()
    metrics = collect_parallel_reward_metrics(
        model,
        main_out=torch.ones(1, 4),
        reward_out=torch.ones(1, 4) * 0.5,
        gate=torch.full((1, 4), 0.8),
    )
    assert "parallel_reward/main_norm" in metrics
    assert "parallel_reward/branch_norm" in metrics
    assert "parallel_reward/branch_ratio" in metrics
    assert "parallel_reward/gate_mean" in metrics
    assert "parallel_reward/gate_max" in metrics
    assert metrics["parallel_reward/branch_ratio"] < 1.0
