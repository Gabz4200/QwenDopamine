"""Behavioral tests for ConfigAdapter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qwendopamine.models.config_adapter import (
    _DEFAULT_HIDDEN_SIZE,
    _DEFAULT_MAX_POSITION_EMBEDDINGS,
    _DEFAULT_NUM_HIDDEN_LAYERS,
    _DEFAULT_VOCAB_SIZE,
    ConfigAdapter,
)


@dataclass
class DummyConfig:
    hidden_size: int = 128
    vocab_size: int = 1000
    max_position_embeddings: int = 2048
    num_hidden_layers: int = 4


class OnlyAliasesConfig:
    n_embd: int = 256
    n_layer: int = 8


class EmptyConfig:
    pass


def test_when_all_fields_present_then_returns_own_values() -> None:
    config = DummyConfig()
    adapter = ConfigAdapter(config, family="test")
    assert adapter.hidden_size == 128
    assert adapter.vocab_size == 1000
    assert adapter.max_position_embeddings == 2048
    assert adapter.num_hidden_layers == 4


def test_when_using_legacy_aliases_then_returns_alias_values() -> None:
    config = OnlyAliasesConfig()
    adapter = ConfigAdapter(config, family="legacy")
    assert adapter.hidden_size == 256
    assert adapter.num_hidden_layers == 8


def test_when_config_is_empty_then_returns_defaults() -> None:
    adapter = ConfigAdapter(EmptyConfig(), family="empty")
    assert adapter.hidden_size == _DEFAULT_HIDDEN_SIZE
    assert adapter.vocab_size == _DEFAULT_VOCAB_SIZE
    assert adapter.max_position_embeddings == _DEFAULT_MAX_POSITION_EMBEDDINGS
    assert adapter.num_hidden_layers == _DEFAULT_NUM_HIDDEN_LAYERS


def test_when_unknown_attribute_then_forwards_to_wrapped_config() -> None:
    config = DummyConfig()
    adapter = ConfigAdapter(config, family="test")
    assert adapter.hidden_size == 128
    with pytest.raises(AttributeError):
        _ = adapter.nonexistent_field


def test_when_mixed_attributes_then_prefers_primary_over_alias() -> None:
    @dataclass
    class MixedConfig:
        hidden_size: int = 64
        n_embd: int = 999

    adapter = ConfigAdapter(MixedConfig(), family="mixed")
    assert adapter.hidden_size == 64


def test_when_repr_called_then_includes_family_and_wrapped_config() -> None:
    config = DummyConfig()
    adapter = ConfigAdapter(config, family="test")
    repr_str = repr(adapter)
    assert "family='test'" in repr_str
    assert "DummyConfig" in repr_str
