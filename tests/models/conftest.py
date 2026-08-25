"""Shared test fixtures for QwenDopamine test suite."""

from __future__ import annotations

import pytest
import torch

from qwendopamine.models.qwen35 import Qwen3_5TextConfig


@pytest.fixture
def tiny_qwen35_config() -> Qwen3_5TextConfig:
    r"""Fixture providing a fast, minimal Qwen3.5 configuration."""
    return Qwen3_5TextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


@pytest.fixture
def device() -> torch.device:
    """Fixture providing the default compute device."""
    return torch.device("cpu")
