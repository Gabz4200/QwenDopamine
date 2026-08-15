from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from qwendopamine.utils import get_model_device


@dataclass
class TrainConfig:
    r"""Training configuration dataclass.

    Attributes:
        max_steps (int): maximum number of training steps. Default: ``100000``.
        grad_accum_steps (int): number of gradient accumulation steps. Default: ``1``.
        max_grad_norm (float): gradient clipping threshold. Default: ``1.0``.
        mixed_precision (str): mixed precision mode. Accepted values: ``"bf16"``
            or ``"fp16"``. Default: ``"bf16"``.
    """

    max_steps: int = 100000
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"


class TrainingLoop:
    r"""Minimal training loop with gradient accumulation and mixed precision.

    Args:
        model (nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer.
        scheduler (LRScheduler): learning rate scheduler.
        config (TrainConfig): training configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        config: TrainConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.scaler = GradScaler(
            get_model_device(model).type, enabled=config.mixed_precision == "fp16"
        )
        self.mixed_precision = config.mixed_precision
        self.global_step = 0

    def run(self, train_loader: Any) -> None:
        r"""Run training for up to ``config.max_steps`` optimizer steps.

        Args:
            train_loader (Any): iterable yielding training batches.
        """
        self.model.train()
        accum = 0
        for accum, batch in enumerate(train_loader, start=1):
            batch = self._move_to_device(self.model, batch)
            autocast_device = get_model_device(self.model).type
            with torch.autocast(
                device_type=autocast_device,
                dtype=torch.bfloat16
                if self.mixed_precision == "bf16"
                else torch.float16,
                enabled=self.mixed_precision in ("bf16", "fp16"),
            ):
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()  # type: ignore[arg-type]

            if accum % self.config.grad_accum_steps == 0:
                self._optimizer_step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.global_step >= self.config.max_steps:
                    break

        if accum > 0 and accum % self.config.grad_accum_steps != 0:
            self._optimizer_step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

    def _optimizer_step(self) -> None:
        r"""Unscale gradients, clip, and perform an optimizer step."""
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    @staticmethod
    def _move_to_device(model: nn.Module, batch: Any) -> Any:
        r"""Recursively move tensors in a batch to the model's device.

        Args:
            model (nn.Module): model whose device the batch is moved to.
            batch (Any): tensor, dict of tensors, or nested structure.

        Returns:
            Any: batch with tensors moved to the model's device.
        """
        device = get_model_device(model)
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, dict):
            return {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        return batch
