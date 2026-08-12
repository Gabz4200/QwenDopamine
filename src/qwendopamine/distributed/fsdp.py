from __future__ import annotations

from typing import Any

import torch


def wrap_with_fsdp(model: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return FSDP(model, **kwargs)
