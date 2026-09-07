"""Domain ports for model interactions.

These protocols define the hexagonal boundary between the QwenDopamine
domain model and concrete adapters such as Hugging Face integration,
training loops, and evaluation utilities.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelPort(Protocol):
    """Minimal contract for training and evaluation adapters."""

    def eval(self) -> ModelPort: ...
    def train(self, mode: bool = True) -> ModelPort: ...
    def parameters(self) -> Any: ...
    def to(self, *args: Any, **kwargs: Any) -> ModelPort: ...
    def __call__(self, **kwargs: Any) -> Any: ...
    def generate(self, **kwargs: Any) -> Any: ...
    @property
    def config(self) -> Any: ...


@runtime_checkable
class ModelFactoryPort(Protocol):
    """Contract for loading/constructing model instances."""

    def from_pretrained(self, repo_id: str, **kwargs: Any) -> ModelPort: ...
    def from_config(self, config: Any, **kwargs: Any) -> ModelPort: ...
