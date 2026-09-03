r"""Learning-rate schedulers and linear warmup wrappers."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    warmup_steps: int = 2000,
    min_lr: float = 1e-5,
    max_steps: int = 100000,
) -> LRScheduler:
    r"""build_scheduler(optimizer: Optimizer, name: str, warmup_steps: int = 2000, min_lr: float = 1e-5, max_steps: int = 100000) -> LRScheduler

    Build a learning-rate scheduler with linear warmup wrapper.

    Args:
        optimizer (Optimizer): Optimizer to schedule.
        name (str): Scheduler name (``"cosine"``).
        warmup_steps (int): Warmup steps. Default: ``2000``.
        min_lr (float): Minimum learning rate. Default: ``1e-5``.
        max_steps (int): Total training steps. Default: ``100000``.

    Returns:
        LRScheduler: A :class:`LinearWarmupScheduler` for ``name="cosine"``.

    Raises:
        KeyError: If ``name`` is not recognised.
    """
    if name == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(max_steps - warmup_steps, 1), eta_min=min_lr
        )
        return LinearWarmupScheduler(optimizer, base_scheduler, warmup_steps, min_lr)
    raise KeyError(f"Unknown scheduler: {name}")


class LinearWarmupScheduler(LRScheduler):
    r"""LinearWarmupScheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, min_lr: float, base_scheduler: LRScheduler) -> None

    Linearly warm up the learning rate, then delegate to a base scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        warmup_steps (int): Number of warmup steps.
        min_lr (float): Minimum learning rate floor.
        base_scheduler (LRScheduler): Scheduler to delegate to after warmup.
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
        # step_count must exist before super().__init__ because LRScheduler calls self.step().
        self.step_count = 0
        super().__init__(optimizer)
        # Reset after init: LRScheduler may call self.step() during setup.
        self.step_count = 0

    def step(self, epoch: int | None = None) -> None:
        r"""step(epoch: int | None = None) -> None

        Advance the scheduler by one step.

        During warmup, sets a linear scale on each param group. After
        warmup, delegates to the base scheduler.

        Args:
            epoch (int | None): Epoch for the base scheduler. Default: ``None``.

        Returns:
            None
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

        Return scheduler state preserving warmup progress and base scheduler.

        Returns:
            dict[str, Any]: State with keys ``base_scheduler``,
            ``step_count``, ``warmup_steps``, ``min_lr``.
        """
        return {
            "base_scheduler": self.base_scheduler.state_dict(),
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""load_state_dict(state_dict: dict[str, Any]) -> None

        Restore scheduler state from a state dict.

        Args:
            state_dict (dict[str, Any]): State dict from :meth:`state_dict`.

        Returns:
            None
        """
        if "base_scheduler" in state_dict:
            self.base_scheduler.load_state_dict(state_dict["base_scheduler"])
            self.step_count = state_dict.get("step_count", self.step_count)
            self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
            self.min_lr = state_dict.get("min_lr", self.min_lr)
        else:
            self.base_scheduler.load_state_dict(state_dict)

        if self.step_count <= self.warmup_steps and self.warmup_steps > 0:
            scale = self.step_count / max(self.warmup_steps, 1)
            for group in self.optimizer.param_groups:
                initial_lr = float(
                    group.get("initial_lr", group.get("lr", self.min_lr)) or self.min_lr
                )
                group["lr"] = initial_lr * scale


__all__ = [
    "LinearWarmupScheduler",
    "build_scheduler",
]
