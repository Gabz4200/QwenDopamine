"""Training loop and utilities."""

from qwendopamine.training.freezing import (
    freeze_module,
    set_trainable,
    trainable_parameters,
    unfreeze_module,
    validate_unfreeze_phases,
)
from qwendopamine.training.loop import TrainingLoop
from qwendopamine.training.metrics import MetricTracker
from qwendopamine.training.parallel_reward import (
    collect_parallel_reward_metrics,
    maybe_warn_branch_ratio,
)
from qwendopamine.training.schedules import build_scheduler

__all__ = [
    "MetricTracker",
    "TrainingLoop",
    "build_scheduler",
    "collect_parallel_reward_metrics",
    "freeze_module",
    "maybe_warn_branch_ratio",
    "set_trainable",
    "trainable_parameters",
    "unfreeze_module",
    "validate_unfreeze_phases",
]
