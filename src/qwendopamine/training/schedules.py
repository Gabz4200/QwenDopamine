from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_scheduler(optimizer: Optimizer, name: str, warmup_steps: int = 2000, min_lr: float = 1e-5, **kwargs: Any) -> LRScheduler:
    if name == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=min_lr)
        return LinearWarmupScheduler(optimizer, base_scheduler, warmup_steps, min_lr)
    raise KeyError(f"Unknown scheduler: {name}")


class LinearWarmupScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, base_scheduler: LRScheduler, warmup_steps: int, min_lr: float) -> None:
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_count = 0
        super().__init__(optimizer)

    def step(self, epoch: int | None = None) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(self.warmup_steps, 1)
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
            return
        self.base_scheduler.step(epoch)

    def state_dict(self) -> dict[str, Any]:  # pragma: no cover - placeholder
        return self.base_scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # pragma: no cover - placeholder
        self.base_scheduler.load_state_dict(state_dict)
