r"""Parameter freezing and unfreezing utilities for PyTorch modules."""

from __future__ import annotations

from torch import nn


def set_trainable(module: nn.Module, enabled: bool) -> None:
    r"""set_trainable(module, enabled) -> None

    Sets the ``requires_grad`` gradient tracking flag on all parameters of a module.

    Args:
        module (nn.Module): Target PyTorch module.
        enabled (bool): If ``True``, enables gradient calculation; if ``False``, freezes parameters.

    Examples::

        >>> module = nn.Linear(10, 5)
        >>> set_trainable(module, False)
        >>> any(p.requires_grad for p in module.parameters())
        False
    """
    for param in module.parameters():
        param.requires_grad_(enabled)


def freeze_module(module: nn.Module) -> None:
    r"""freeze_module(module) -> None

    Freezes all parameters of a module by setting ``requires_grad=False``.

    Args:
        module (nn.Module): Target PyTorch module.

    Examples::

        >>> module = nn.Linear(10, 5)
        >>> freeze_module(module)
        >>> any(p.requires_grad for p in module.parameters())
        False
    """
    set_trainable(module, False)


def unfreeze_module(module: nn.Module) -> None:
    r"""unfreeze_module(module) -> None

    Unfreezes all parameters of a module by setting ``requires_grad=True``.

    Args:
        module (nn.Module): Target PyTorch module.

    Examples::

        >>> module = nn.Linear(10, 5)
        >>> freeze_module(module)
        >>> unfreeze_module(module)
        >>> all(p.requires_grad for p in module.parameters())
        True
    """
    set_trainable(module, True)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    r"""trainable_parameters(model) -> list[nn.Parameter]

    Returns a list of all parameters in a model that have ``requires_grad=True``.

    Args:
        model (nn.Module): Source PyTorch model.

    Returns:
        list[nn.Parameter]: List of trainable parameters.

    Examples::

        >>> model = nn.Linear(10, 5)
        >>> params = trainable_parameters(model)
        >>> len(params)
        2
    """
    return [param for param in model.parameters() if param.requires_grad]


__all__ = [
    "freeze_module",
    "set_trainable",
    "trainable_parameters",
    "unfreeze_module",
]
