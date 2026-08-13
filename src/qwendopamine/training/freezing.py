from __future__ import annotations

import torch


def set_trainable(module: torch.nn.Module, enabled: bool) -> None:
    r"""Set ``requires_grad`` on all parameters of a module.

    Args:
        module (torch.nn.Module): target module.
        enabled (bool): if ``True``, parameters become trainable.
    """
    for param in module.parameters():
        param.requires_grad_(enabled)


def freeze_module(module: torch.nn.Module) -> None:
    r"""Freeze all parameters of a module.

    Args:
        module (torch.nn.Module): target module.
    """
    set_trainable(module, False)


def unfreeze_module(module: torch.nn.Module) -> None:
    r"""Unfreeze all parameters of a module.

    Args:
        module (torch.nn.Module): target module.
    """
    set_trainable(module, True)


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    r"""List trainable parameters from a model.

    Args:
        model (torch.nn.Module): source model.

    Returns:
        list[torch.nn.Parameter]: parameters with ``requires_grad=True``.
    """
    return [param for param in model.parameters() if param.requires_grad]
