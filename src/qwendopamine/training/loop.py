from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler

from qwen_dopamine.training.schedules import Scheduler


@dataclass
class TrainConfig:
    max_steps: int = 100000
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"


class TrainingLoop:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Scheduler, config: TrainConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.scaler = GradScaler(enabled=config.mixed_precision == "fp16")
        self.global_step = 0

    def run(self, train_loader: Any) -> None:
        self.model.train()
        accumulator = 0

        for batch in train_loader:
            batch = self._move_to_device(batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.mixed_precision == "bf16"):
                outputs = self.model(**batch)
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()
            accumulator += 1

            if accumulator % self.config.grad_accum_steps == 0:
                self._optimizer_step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    @staticmethod
    def _move_to_device(batch: Any, device: torch.device | str = "cuda") -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch
