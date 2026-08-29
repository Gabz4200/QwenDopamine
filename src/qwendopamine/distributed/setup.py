r"""Distributed training process group setup and teardown utilities."""

from __future__ import annotations

import os

import torch


def _default_backend() -> str:
    r"""Pick the appropriate PyTorch distributed backend based on GPU availability."""
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def init_distributed() -> tuple[int, int, int]:
    r"""Initialize the PyTorch distributed process group if ``WORLD_SIZE > 1``.

    If ``RANK``, ``WORLD_SIZE``, or ``LOCAL_RANK`` are not set, they are
    defaulted to ``"0"``, ``"1"``, and ``"0"`` respectively via
    ``os.environ.setdefault``, mutating the process environment as a side
    effect. Pass these variables explicitly in multi-process launches.
    """
    if not os.environ.get("RANK"):
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if world_size > 1:
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(
                backend=_default_backend(), rank=rank, world_size=world_size
            )

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    r"""Destroy the PyTorch distributed process group if currently initialized."""
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "cleanup_distributed",
    "init_distributed",
]
