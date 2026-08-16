r"""Learning-rate schedulers and linear warmup wrappers."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_scheduler(
    optimizer: Optimizer, name: str, warmup_steps: int = 2000, min_lr: float = 1e-5
) -> LRScheduler:
    r"""build_scheduler(optimizer, name, warmup_steps=2000, min_lr=1e-5) -> LRScheduler

    Constructs a learning-rate scheduler with linear warmup wrapper.

    Args:
        optimizer (Optimizer): PyTorch optimizer instance to schedule.
        name (str): Scheduler name string. Supported values: ``"cosine"``.
        warmup_steps (int, optional): Number of linear warmup steps. Default: ``2000``.
        min_lr (float, optional): Minimum learning rate floor after decay. Default: ``1e-5``.

    Returns:
        LRScheduler: Wrapped learning rate scheduler with warmup logic.

    Raises:
        KeyError: If ``name`` is not a supported scheduler type.
    """
    if name == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100000, eta_min=min_lr
        )
        return LinearWarmupScheduler(optimizer, base_scheduler, warmup_steps, min_lr)
    raise KeyError(f"Unknown scheduler: {name}")


class LinearWarmupScheduler(LRScheduler):
    r"""LinearWarmupScheduler(optimizer, base_scheduler, warmup_steps, min_lr)

    Linearly increases learning rate from zero to initial base learning rate over initial warmup steps
    before delegating scheduling to a base scheduler.

    Args:
        optimizer (Optimizer): PyTorch optimizer instance.
        base_scheduler (LRScheduler): Base scheduler taking over after warmup.
        warmup_steps (int): Total number of linear warmup steps.
        min_lr (float): Minimum learning rate floor.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: LRScheduler,
        warmup_steps: int,
        min_lr: float,
    ) -> None:
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_count = 0
        super().__init__(optimizer)
        self.step_count = 0

    def step(self, epoch: int | None = None) -> None:
        r"""step(epoch=None) -> None

        Steps the scheduler state forward.

        Args:
            epoch (int, optional): Epoch index. Default: ``None``.
        """
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(self.warmup_steps, 1)
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
            return
        self.base_scheduler.step(epoch)

    def state_dict(self) -> dict[str, Any]:
        r"""state_dict() -> dict[str, Any]

        Returns state dictionary of the wrapped base scheduler.

        Returns:
            dict[str, Any]: Base scheduler state dict.
        """
        return self.base_scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""load_state_dict(state_dict) -> None

        Loads state dictionary into the wrapped base scheduler.

        Args:
            state_dict (dict[str, Any]): Target state dictionary.
        """
        self.base_scheduler.load_state_dict(state_dict)


__all__ = [
    "LinearWarmupScheduler",
    "build_scheduler",
]
