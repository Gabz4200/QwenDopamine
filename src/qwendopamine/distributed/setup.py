r"""Distributed training process group setup and teardown utilities."""

from __future__ import annotations

import os

import torch


def _default_backend() -> str:
    r"""_default_backend() -> str

    Picks the appropriate PyTorch distributed backend string ("nccl" or "gloo") based on GPU availability.
    """
    if torch.cuda.is_available():
        return "nccl"
    if torch.backends.mps.is_available():
        return "gloo"
    return "gloo"


def init_distributed() -> tuple[int, int, int]:
    r"""init_distributed() -> (int, int, int)

    Initializes the PyTorch distributed process group if ``WORLD_SIZE > 1``.

    Defaults environment variables ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` to ``0``/``1``/``0``
    for single-process execution.

    Returns:
        tuple[int, int, int]: Tuple containing ``(rank, world_size, local_rank)``.
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

        if not dist.is_initialized():
            dist.init_process_group(
                backend=_default_backend(), rank=rank, world_size=world_size
            )

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    r"""cleanup_distributed() -> None

    Destroys the PyTorch distributed process group if currently initialized.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "cleanup_distributed",
    "init_distributed",
]
