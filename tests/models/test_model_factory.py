"""Behavioral tests for model_factory helpers."""

from __future__ import annotations

from torch import nn

from qwendopamine.models.infinidopamine.configs import InfiniDopamineTextConfig
from qwendopamine.models.model_factory import (
    _resolve_model_family,
    _unwrap_text_config,
    build_model,
    build_reference_model,
)
from qwendopamine.models.qwen35.configs import Qwen3_5TextConfig


def test_unwrap_text_config_returns_text_config_when_present() -> None:
    text_config = Qwen3_5TextConfig()
    composite = type("CompositeConfig", (), {"text_config": text_config})()
    assert _unwrap_text_config(composite) is text_config


def test_unwrap_text_config_returns_original_when_no_text_config() -> None:
    config = Qwen3_5TextConfig()
    assert _unwrap_text_config(config) is config


def test_resolve_model_family_with_infinidopamine_text_config() -> None:
    config = InfiniDopamineTextConfig()
    family, _ = _resolve_model_family(config)
    assert family == "infinidopamine"


def test_resolve_model_family_with_qwen35_text_config() -> None:
    config = Qwen3_5TextConfig()
    family, _ = _resolve_model_family(config)
    assert family == "qwen35"


def test_resolve_model_family_with_unknown_model_type_returns_unknown() -> None:
    config = type("UnknownConfig", (), {"model_type": "unknown"})()
    family, _ = _resolve_model_family(config)
    assert family == "unknown"


def test_build_model_with_qwen35_text_config_returns_qwen35_model() -> None:
    config = Qwen3_5TextConfig(
        hidden_size=32,
        vocab_size=100,
        max_position_embeddings=64,
        num_hidden_layers=2,
    )
    model = build_model(config)
    assert type(model).__name__ == "Qwen3_5ForCausalLM"


def test_build_model_with_infinidopamine_text_config_returns_infini_model() -> None:
    config = InfiniDopamineTextConfig(
        hidden_size=32,
        vocab_size=100,
        max_position_embeddings=64,
        num_hidden_layers=2,
    )
    model = build_model(config)
    assert type(model).__name__ == "InfiniDopamineForCausalLM"


def test_build_reference_model_returns_nn_module() -> None:
    config = Qwen3_5TextConfig(
        hidden_size=32,
        vocab_size=100,
        max_position_embeddings=64,
        num_hidden_layers=2,
    )
    model = build_reference_model(config, device_map="cpu")
    assert isinstance(model, nn.Module)
