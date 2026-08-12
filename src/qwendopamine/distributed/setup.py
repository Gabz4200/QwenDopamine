from __future__ import annotations

import os


def init_distributed() -> tuple[int, int, int]:
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
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
