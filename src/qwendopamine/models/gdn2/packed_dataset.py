# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

from __future__ import annotations

from typing import Any

from torch.utils.data import IterableDataset


class PackedDataset(IterableDataset[Any]):
    """PackedDataset placeholder for legacy sequence packing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()


__all__ = ["PackedDataset"]
