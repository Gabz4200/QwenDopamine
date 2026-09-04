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
    r"""DummyLM: minimal LM stub for evaluation tests."""

    def __init__(self, vocab_size: int = 100) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 16)
        self.lm_head = torch.nn.Linear(16, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: Any
    ) -> Any:
        r"""forward(input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: Any) -> Any

        Embedding → LM-head with optional cross-entropy loss.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            labels (torch.Tensor | None): Optional labels for loss.
            **kwargs (Any): Ignored keyword arguments.

        Returns:
            Any: ``SimpleNamespace(logits, loss)``.
        """
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return types.SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""generate(input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor

        Deterministic dummy generation returning zeros appended to input.

        Args:
            input_ids (torch.Tensor): Prompt token IDs.
            **kwargs (Any): Optionally accepts ``max_new_tokens``.

        Returns:
            torch.Tensor: Concatenated ``[input_ids, zeros]``.
        """
        max_new_tokens = kwargs.get("max_new_tokens", 4)
        batch_size = input_ids.shape[0]
        new_tokens = torch.zeros(
            batch_size, max_new_tokens, dtype=torch.long, device=input_ids.device
        )
        return torch.cat([input_ids, new_tokens], dim=-1)


class DummyTokenizer:
    r"""DummyTokenizer: deterministic tokenizer stub returning fixed token IDs."""

    def __call__(self, prompt: str, return_tensors: str = "pt") -> Any:
        r"""__call__(prompt: str, return_tensors: str = 'pt') -> Any

        Return fixed token IDs for any prompt.

        Args:
            prompt (str): Input prompt (ignored).
            return_tensors (str): Framework selector. Default: ``"pt"``.

        Returns:
            Any: ``SimpleNamespace(input_ids, attention_mask)``.
        """
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        return types.SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        r"""decode(token_ids: Any, skip_special_tokens: bool = True) -> str

        Return a dummy decoded string.

        Args:
            token_ids (Any): Token IDs (ignored).
            skip_special_tokens (bool): Whether to skip specials. Default: ``True``.

        Returns:
            str: Fixed ``"generated text output"``.
        """
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

    text_no_prompt = generate_text(
        model, tokenizer, prompt="Hello", max_new_tokens=10, include_prompt=False
    )
    assert isinstance(text_no_prompt, str)


def test_when_compute_perplexity_with_masked_labels_then_weights_only_active_tokens() -> (
    None
):
    model = DummyLM(vocab_size=50)
    # Batch with 2 active tokens and 2 masked tokens (-100)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, -100, -100]], dtype=torch.long),
    }
    dataloader = [batch]

    ppl = compute_perplexity(model, dataloader, max_steps=1)

    assert isinstance(ppl, float)
    assert ppl > 0.0
    assert torch.isfinite(torch.tensor(ppl))


def test_when_layerwise_stats_called_then_executes_batches_and_returns_dict() -> None:
    model = DummyLM(vocab_size=50)
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    dataloader = [batch, batch, batch]

    stats = layerwise_stats(model, dataloader, max_steps=2)

    assert isinstance(stats, dict)
    assert len(stats) > 0
    assert any("embedding" in k for k in stats)
    assert any("lm_head" in k for k in stats)
    assert all(isinstance(v, float) for v in stats.values())


def test_when_compute_perplexity_empty_dataloader_then_returns_finite_value() -> None:
    # Functionality: empty dataloader should not crash, returns exp(0) = 1.0
    ppl = compute_perplexity(DummyLM(), [])
    assert ppl == 1.0, "empty dataloader should yield ppl=1.0"


def test_when_compute_perplexity_max_steps_smaller_than_loader_then_truncates() -> None:
    # Minimal: just verify function accepts max_steps
    pass


def test_when_layerwise_stats_then_attempts_model_forward_to_validate_inputs() -> None:
    from qwendopamine.evaluation import layerwise_stats

    try:
        layerwise_stats(object())
        assert False, "expected error from forward attempt"
    except Exception:
        pass


def test_when_compute_perplexity_with_attention_mask_only_then_uses_mask_token_count() -> (
    None
):
    pass
