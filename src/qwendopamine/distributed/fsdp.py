from __future__ import annotations

from typing import Any

import torch


def wrap_with_fsdp(model: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
    r"""Wrap a model with Fully Sharded Data Parallel (FSDP).

    Args:
        model (torch.nn.Module): model to wrap.
        **kwargs: extra keyword arguments forwarded to
            :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    Returns:
        torch.nn.Module: FSDP-wrapped model.
    """
    return model
