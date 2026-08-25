r"""Parameter freezing and unfreezing utilities for PyTorch modules."""

from __future__ import annotations

import logging

from torch import nn

logger = logging.getLogger(__name__)


def validate_unfreeze_phases(
    model: nn.Module, unfreeze_phases: list[dict[str, object]]
) -> list[dict[str, object]]:
    r"""validate_unfreeze_phases(model, unfreeze_phases) -> list[dict[str, object]]

    Validate that module names in unfreeze phases exist in the model.
    Logs warnings for non-existent modules but does not raise, allowing
    training to proceed with a clear diagnostic.

    Args:
        model (nn.Module): The model to validate against.
        unfreeze_phases (list[dict]): List of phase dicts, each with a
            ``"modules"`` key containing module name strings.

    Returns:
        list[dict]: The validated unfreeze phases (unchanged).
    """
    model_module_names = {name for name, _ in model.named_modules()}
    for phase in unfreeze_phases:
        modules = phase.get("modules", [])
        if not isinstance(modules, list):
            continue
        for module_name in modules:
            if not isinstance(module_name, str):
                continue
            if module_name not in model_module_names:
                suggestion = _suggest_similar(module_name, model_module_names)
                hint = f" Did you mean: {suggestion}?" if suggestion else ""
                logger.warning(
                    "Unfreeze phase references module '%s' which does not exist "
                    "in the model. This will be a silent no-op.%s",
                    module_name,
                    hint,
                )
    return unfreeze_phases


def _suggest_similar(
    name: str, known_names: set[str], threshold: int = 3
) -> str | None:
    """Return the closest matching module name if edit distance is small."""
    best: str | None = None
    best_dist = threshold + 1
    for known in known_names:
        dist = _edit_distance(name, known)
        if dist < best_dist:
            best_dist = dist
            best = known
    return best


def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


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
    "validate_unfreeze_phases",
]
