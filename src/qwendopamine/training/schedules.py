from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_scheduler(optimizer: Optimizer, name: str, warmup_steps: int = 2000, min_lr: float = 1e-5, **kwargs: Any) -> LRScheduler:
    r"""Build a learning-rate scheduler with linear warmup.

    Currently supports ``"cosine"`` with linear warmup.

    Args:
        optimizer (Optimizer): wrapped optimizer.
        name (str): scheduler name. Accepted values: ``"cosine"``.
        warmup_steps (int): number of warmup steps. Default: ``2000``.
        min_lr (float): minimum learning rate after decay. Default: ``1e-5``.
        **kwargs: extra keyword arguments reserved for future scheduler types.

    Returns:
        LRScheduler: wrapped scheduler with warmup.

    Raises:
        KeyError: if ``name`` is not supported.
    """
    if name == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=min_lr)
        return LinearWarmupScheduler(optimizer, base_scheduler, warmup_steps, min_lr)
    raise KeyError(f"Unknown scheduler: {name}")


class LinearWarmupScheduler(LRScheduler):
    r"""Linear warmup scheduler wrapping a base LR scheduler.

    During the first ``warmup_steps`` calls, the learning rate scales linearly
    from ``0`` to ``initial_lr``. Afterwards, the base scheduler takes over.

    Args:
        optimizer (Optimizer): wrapped optimizer.
        base_scheduler (LRScheduler): scheduler to delegate to after warmup.
        warmup_steps (int): number of warmup steps.
        min_lr (float): minimum learning rate floor.
    """
    def __init__(self, optimizer: Optimizer, base_scheduler: LRScheduler, warmup_steps: int, min_lr: float) -> None:
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_count = 0
        super().__init__(optimizer)

    def step(self, epoch: int | None = None) -> None:
        r"""Step the scheduler.

        Args:
            epoch (int, optional): unused; kept for compatibility with
                :class:`torch.optim.lr_scheduler.LRScheduler`.
        """
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(self.warmup_steps, 1)
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
            return
        self.base_scheduler.step(epoch)

    def state_dict(self) -> dict[str, Any]:  # pragma: no cover - placeholder
        r"""Return the base scheduler state dict.

        Returns:
            dict[str, Any]: state dict.
        """
        return self.base_scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # pragma: no cover - placeholder
        r"""Load state into the base scheduler.

        Args:
            state_dict (dict[str, Any]): state dict to load.
        """
        self.base_scheduler.load_state_dict(state_dict)
