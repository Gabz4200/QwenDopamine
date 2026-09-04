"""Behavioral tests for utility functions."""

from __future__ import annotations

import torch
from torch import nn

from qwendopamine.distributed.setup import (
    _default_backend,
    cleanup_distributed,
    init_distributed,
)
from qwendopamine.utils import get_model_device, move_to_device


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


def test_when_move_to_device_with_tensor_then_moves_tensor() -> None:
    tensor = torch.randn(2, 3)
    device = torch.device("cpu")
    result = move_to_device(tensor, device)
    assert result.device == device


def test_when_move_to_device_with_dict_then_maps_values() -> None:
    batch = {"a": torch.randn(2, 3), "b": torch.randn(4)}
    device = torch.device("cpu")
    result = move_to_device(batch, device)
    assert isinstance(result, dict)
    assert result["a"].device == device
    assert result["b"].device == device


def test_when_move_to_device_with_list_then_moves_items() -> None:
    batch = [torch.randn(2, 3), torch.randn(4)]
    device = torch.device("cpu")
    result = move_to_device(batch, device)
    assert isinstance(result, list)
    assert result[0].device == device
    assert result[1].device == device


def test_when_move_to_device_with_tuple_then_preserves_type() -> None:
    batch = (torch.randn(2, 3), torch.randn(4))
    device = torch.device("cpu")
    result = move_to_device(batch, device)
    assert isinstance(result, tuple)
    assert result[0].device == device
    assert result[1].device == device


def test_when_move_to_device_with_non_tensor_then_passes_through() -> None:
    batch = {"a": 1, "b": "hello"}
    device = torch.device("cpu")
    result = move_to_device(batch, device)
    assert result == batch


def test_when_move_to_device_dict_has_none_value_then_none_preserved() -> None:
    import torch

    from qwendopamine.utils import move_to_device

    d = {"a": torch.tensor([1.0]), "b": None}
    result = move_to_device(d, torch.device("cpu"))
    assert result["b"] is None


def test_when_move_to_device_list_has_none_then_none_preserved() -> None:
    import torch

    from qwendopamine.utils import move_to_device

    l = [torch.tensor([1.0]), None]
    result = move_to_device(l, torch.device("cpu"))
    assert result[1] is None
