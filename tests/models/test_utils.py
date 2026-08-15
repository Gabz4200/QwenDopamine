"""Behavioral tests for utility functions."""

from __future__ import annotations

import torch
from torch import nn

from qwendopamine.distributed.setup import (
    _default_backend,
    cleanup_distributed,
    init_distributed,
)
from qwendopamine.utils import get_model_device


def test_when_model_has_parameters_then_returns_parameter_device() -> None:
    model = nn.Linear(10, 10)
    device = get_model_device(model)
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_when_model_has_no_parameters_then_falls_back_to_cpu() -> None:
    empty_model = nn.Module()
    device = get_model_device(empty_model)
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_when_init_distributed_in_single_process_then_returns_zero_rank() -> None:
    rank, world_size, local_rank = init_distributed()
    assert rank == 0
    assert world_size == 1
    assert local_rank == 0


def test_when_default_backend_queried_then_returns_valid_string() -> None:
    backend = _default_backend()
    assert backend in ("nccl", "gloo")


def test_when_cleanup_distributed_called_then_executes_without_error() -> None:
    cleanup_distributed()
