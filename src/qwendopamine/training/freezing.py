from __future__ import annotations

import torch


def set_trainable(module: torch.nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def freeze_module(module: torch.nn.Module) -> None:
    set_trainable(module, False)


def unfreeze_module(module: torch.nn.Module) -> None:
    set_trainable(module, True)


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]
