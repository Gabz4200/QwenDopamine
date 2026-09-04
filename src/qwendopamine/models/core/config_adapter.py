"""Config adapters that normalize field access across model families."""

from __future__ import annotations

from typing import Any

_DEFAULT_HIDDEN_SIZE = 768
_DEFAULT_VOCAB_SIZE = 151936
_DEFAULT_MAX_POSITION_EMBEDDINGS = 32768
_DEFAULT_NUM_HIDDEN_LAYERS = 12


class ConfigAdapter:
    r"""Adapter that normalizes field access for model configs.

    This wrapper provides a unified interface for accessing common configuration
    fields across different model families, reducing duplicated ``getattr`` chains
    throughout the codebase.

    Args:
        config (Any): Raw configuration object from any model family.
        family (str): Model family identifier (e.g., ``"infinidopamine"``,
            ``"qwen35"``, ``"gdn2"``).
    """

    def __init__(self, config: Any, family: str) -> None:
        self._config = config
        self.family = family

    @property
    def hidden_size(self) -> int:
        r"""Hidden dimension size with family-specific fallbacks."""
        return getattr(
            self._config,
            "hidden_size",
            getattr(
                self._config,
                "d_model",
                getattr(self._config, "n_embd", _DEFAULT_HIDDEN_SIZE),
            ),
        )

    @property
    def vocab_size(self) -> int:
        r"""Vocabulary size with family-specific fallbacks."""
        return getattr(self._config, "vocab_size", _DEFAULT_VOCAB_SIZE)

    @property
    def max_position_embeddings(self) -> int:
        r"""Maximum sequence length with family-specific fallbacks."""
        return getattr(
            self._config, "max_position_embeddings", _DEFAULT_MAX_POSITION_EMBEDDINGS
        )

    @property
    def num_hidden_layers(self) -> int:
        r"""Number of transformer layers with family-specific fallbacks."""
        return getattr(
            self._config,
            "num_hidden_layers",
            getattr(self._config, "n_layer", _DEFAULT_NUM_HIDDEN_LAYERS),
        )

    def __getattr__(self, name: str) -> Any:
        r"""Forward unknown attributes to the wrapped config."""
        try:
            return getattr(self._config, name)
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self._config).__name__} has no attribute {name!r}"
            ) from exc

    def __repr__(self) -> str:
        return f"ConfigAdapter(family={self.family!r}, config={self._config!r})"
