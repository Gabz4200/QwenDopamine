"""General-purpose reward conditioning building blocks — re-export shim.

The classes are split across per-concern modules:

  - :mod:`.scalers` — :class:`AsinhScaler`, :class:`LearnableSoftsign`
  - :mod:`.fourier` — :class:`LearnableFourierFeatures`
  - :mod:`.film`    — :class:`TokenWiseFiLM`
"""

from __future__ import annotations

import torch

__all__ = [
    "AsinhScaler",
    "LearnableFourierFeatures",
    "LearnableSoftsign",
    "TokenWiseFiLM",
    "broadcast_cond",
]


def broadcast_cond(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    r"""Align ``cond`` to the same rank as ``x`` for broadcasting.

    Review N9: this helper was previously defined as ``_broadcast_cond``
    on each of :class:`AsinhScaler`, :class:`LearnableSoftsign`,
    :class:`LearnableFourierFeatures`, and :class:`TokenWiseFiLM`
    (4 copies, identical). Consolidated here so all four call sites
    share one implementation.
    """
    if cond.dim() == 0:
        cond = cond.unsqueeze(0)
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    if x.dim() == 3 and cond.dim() == 2:
        cond = cond.unsqueeze(1)
    elif x.dim() == 2 and cond.dim() == 3:
        if cond.size(1) != 1:
            raise ValueError(
                "When x has shape (B, D), cond with shape (B, L, C) is only "
                f"valid if L == 1. Got cond.shape={tuple(cond.shape)}."
            )
        cond = cond.squeeze(1)
    elif x.dim() != cond.dim():
        raise ValueError(
            "Unsupported combination of x and cond shapes: "
            f"x={tuple(x.shape)}, cond={tuple(cond.shape)}."
        )
    return cond


from qwendopamine.models.blocks.reward.film import TokenWiseFiLM
from qwendopamine.models.blocks.reward.fourier import LearnableFourierFeatures
from qwendopamine.models.blocks.reward.scalers import AsinhScaler, LearnableSoftsign
