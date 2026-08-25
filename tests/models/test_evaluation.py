"""Behavioral tests for evaluation utilities."""

from __future__ import annotations

import types
from typing import Any

import torch

from qwendopamine.evaluation import (
    compute_perplexity,
    generate_text,
    layerwise_stats,
)


class DummyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 100) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 16)
        self.lm_head = torch.nn.Linear(16, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: Any
    ) -> Any:
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return types.SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        max_new_tokens = kwargs.get("max_new_tokens", 4)
        batch_size = input_ids.shape[0]
        new_tokens = torch.zeros(
            batch_size, max_new_tokens, dtype=torch.long, device=input_ids.device
        )
        return torch.cat([input_ids, new_tokens], dim=-1)


class DummyTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt") -> Any:
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        return types.SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        return "generated text output"


def test_when_compute_perplexity_called_then_returns_finite_float() -> None:
    model = DummyLM(vocab_size=50)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
    }
    dataloader = [batch, batch]

    ppl = compute_perplexity(model, dataloader, max_steps=2)

    assert isinstance(ppl, float)
    assert ppl > 0.0
    assert torch.isfinite(torch.tensor(ppl))


def test_when_generate_text_called_then_returns_decoded_string() -> None:
    model = DummyLM(vocab_size=50)
    tokenizer = DummyTokenizer()

    text = generate_text(model, tokenizer, prompt="Hello", max_new_tokens=10)

    assert isinstance(text, str)
    assert text == "generated text output"


def test_when_layerwise_stats_called_then_executes_batches_and_returns_dict() -> None:
    model = DummyLM(vocab_size=50)
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    dataloader = [batch, batch, batch]

    stats = layerwise_stats(model, dataloader, max_steps=2)

    assert isinstance(stats, dict)
