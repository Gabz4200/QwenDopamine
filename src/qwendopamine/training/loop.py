from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from qwendopamine.utils import get_model_device, move_to_device


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
        r"""run(train_loader: Any) -> None

        Run training for up to ``config.max_steps`` optimizer steps.

        Args:
            train_loader (Any): iterable yielding training batches.

        Raises:
            ValueError: If ``train_loader`` is empty.
        """
        self.model.train()
        accum = 0
        batches = list(train_loader)
        if not batches:
            raise ValueError("train_loader is empty; cannot run training loop.")
        for accum, batch in enumerate(batches, start=1):
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

            loss_tensor: torch.Tensor = (
                loss if isinstance(loss, torch.Tensor) else torch.as_tensor(loss)
            )
            self.scaler.scale(loss_tensor).backward()

            if accum % self.config.grad_accum_steps == 0:
                self._step_optimizer()
                if self.global_step >= self.config.max_steps:
                    break

        if accum > 0 and accum % self.config.grad_accum_steps != 0:
            self._step_optimizer()

    def _step_optimizer(self) -> None:
        r"""Unscale, clip gradients, step optimizer/scheduler, and advance global step."""
        stepped = self._optimizer_step()
        if stepped:
            self.scheduler.step()
            self.global_step += 1
        self.optimizer.zero_grad(set_to_none=True)

    def _optimizer_step(self) -> bool:
        r"""Unscale gradients, clip, and perform an optimizer step.

        Returns:
            bool: True if the optimizer step executed, False if skipped due to non-finite gradients.
        """
        if not self.scaler.is_enabled():
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            return True

        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()
        # If scale decreased, the optimizer step was skipped due to Inf/NaN gradients
        return scale_after >= scale_before

    @staticmethod
    def _move_to_device(model: nn.Module, batch: Any) -> Any:
        r"""Recursively move tensors in a batch to the model's device.

        Args:
            model (nn.Module): model whose device the batch is moved to.
            batch (Any): tensor, dict of tensors, or nested structure.

        Returns:
            Any: batch with tensors moved to the model's device.
        """
        return move_to_device(batch, get_model_device(model))
