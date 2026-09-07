"""Behavioral tests for the distributed setup helpers.

The production path requires launching a real multi-process job; the
behavioural contract we can validate in a single-process test is:

* single-process ``init_distributed`` returns ``(0, 1, 0)`` without
  initialising a process group;
* the function mutates the environment only when RANK is unset;
* ``cleanup_distributed`` is idempotent and never raises when no
  process group is active;
* ``_default_backend`` returns ``"nccl"`` only when CUDA is available
  and ``"gloo"`` otherwise.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import torch

from qwendopamine.distributed import cleanup_distributed, init_distributed
from qwendopamine.distributed.setup import _default_backend


@pytest.fixture(autouse=True)
def _restore_env() -> Iterator[None]:
    """Snapshot the distributed-related env vars so tests stay isolated."""
    saved = {k: os.environ.get(k) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_when_init_distributed_in_single_process_then_returns_zero_rank() -> None:
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(k, None)

    rank, world_size, local_rank = init_distributed()

    assert rank == 0
    assert world_size == 1
    assert local_rank == 0
    # The function should have defaulted the env vars to the single-process values.
    assert os.environ["RANK"] == "0"
    assert os.environ["WORLD_SIZE"] == "1"
    assert os.environ["LOCAL_RANK"] == "0"


def test_when_cleanup_distributed_without_process_group_then_is_noop() -> None:
    """``cleanup_distributed`` must be a no-op when the process group is not
    initialised, and must not raise."""
    cleanup_distributed()
    # Idempotent: second call must also be a no-op.
    cleanup_distributed()


def test_when_default_backend_queried_then_returns_valid_string() -> None:
    backend = _default_backend()
    assert backend in {"nccl", "gloo"}
    if torch.cuda.is_available():
        assert backend == "nccl"
    else:
        assert backend == "gloo"


def test_when_init_distributed_called_twice_then_no_state_leak() -> None:
    """Two back-to-back calls must each return consistent ranks."""
    os.environ.pop("RANK", None)
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("LOCAL_RANK", None)

    first = init_distributed()
    second = init_distributed()

    assert first == second
    assert first == (0, 1, 0)
