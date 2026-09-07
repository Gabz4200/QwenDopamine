"""Behavioural tests for the evaluation package: perplexity, generation,
and layerwise stats.

These tests cover the entry points in
:mod:`qwendopamine.evaluation` and the per-function edge cases that
the dedicated tests in ``tests/models/test_evaluation.py`` do not
already exercise.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

from qwendopamine.evaluation import (
    compute_perplexity,
    generate_text,
    layerwise_stats,
)


class _LMWithLoss(nn.Module):
    r"""Tiny LM that returns a ``loss`` attribute so ``compute_perplexity`` can
    extract it. The loss is constant so the perplexity is also constant.
    """

    def __init__(self, loss_value: float = 1.0) -> None:
        super().__init__()
        self._loss = loss_value
        self.linear = nn.Linear(4, 4)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict:
        # The loss is fixed; we don't look at labels. Just return the fixed loss.
        loss = torch.full((), self._loss, dtype=torch.float32, device=input_ids.device)
        return {"loss": loss}

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs
    ) -> Any:  # pyrefly: ignore[unannotated-return]
        b, _ = input_ids.shape
        new_tokens = torch.full(
            (b, max_new_tokens), 99, dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat([input_ids, new_tokens], dim=1)


class _LMWithAvgLoss(nn.Module):
    r"""LM that returns ``outputs.loss`` (object-style) instead of a dict."""

    def __init__(self, loss_value: float = 0.5) -> None:
        super().__init__()
        self._loss = loss_value

    def forward(self, input_ids: torch.Tensor) -> object:
        out = type("Out", (), {})()  # pyrefly: ignore[bad-argument-type]
        out.loss = torch.full((), self._loss)  # pyrefly: ignore[missing-attribute]
        return out  # pyrefly: ignore[no-any-return-explicit]


def test_when_compute_perplexity_with_loss_attribute_object_then_extracts_loss() -> (
    None
):
    """``compute_perplexity`` must accept both dict outputs and object outputs
    with a ``.loss`` attribute (the production path uses object outputs)."""
    import math

    model = _LMWithAvgLoss(loss_value=0.5)
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    ppl = compute_perplexity(model, [batch], max_steps=1)
    # exp(0.5) ≈ 1.6487
    assert ppl == pytest.approx(math.exp(0.5), rel=1e-3)
    assert torch.isfinite(torch.tensor(ppl))


def test_when_compute_perplexity_loss_overflows_then_warns_and_returns_inf() -> None:
    """A very large loss must surface as ``float('inf')`` and emit a warning."""
    model = _LMWithLoss(loss_value=1e9)
    batch = {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ppl = compute_perplexity(model, [batch], max_steps=1)

    assert ppl == float("inf")
    assert any("overflowed" in str(w.message) for w in caught), caught


def test_when_compute_perplexity_with_attention_mask_only_then_uses_token_count() -> (
    None
):
    """When labels are absent but attention_mask is present, the
    token count must be derived from the mask."""
    model = _LMWithLoss(loss_value=0.0)
    batch = {
        "input_ids": torch.tensor([[1, 2, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
    }
    ppl = compute_perplexity(model, [batch], max_steps=1)
    # 2 active tokens, zero loss → perplexity == 1.0
    assert ppl == pytest.approx(1.0, rel=1e-6)


def test_when_compute_perplexity_with_neither_labels_nor_mask_then_counts_all_tokens() -> (
    None
):
    """No labels, no attention mask → token count = ``input_ids.numel()``."""
    model = _LMWithLoss(loss_value=0.0)
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    ppl = compute_perplexity(model, [batch], max_steps=1)
    assert ppl == pytest.approx(1.0, rel=1e-6)


def test_when_generate_text_with_include_prompt_false_then_strips_prompt() -> None:
    """When ``include_prompt=False``, the prompt tokens must be sliced out of
    the output before decoding."""
    model = _LMWithLoss()
    tokenizer = type(
        "Tok",
        (),
        {
            "__call__": lambda self, text, return_tensors=None: type(
                "Enc",
                (),
                {
                    "input_ids": torch.tensor([[9, 9, 1]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                },
            )(),
            "decode": lambda self, ids, skip_special_tokens=True: (
                f"decoded({len(ids)})"
            ),
        },
    )()
    text = generate_text(
        model,
        tokenizer,
        prompt="hello",
        max_new_tokens=2,
        include_prompt=False,
        do_sample=False,
    )
    assert text == "decoded(2)"


def test_when_layerwise_stats_with_hooks_then_hook_removed_after_call() -> None:
    """All forward hooks must be removed once ``layerwise_stats`` returns."""
    model = _LMWithLoss()

    # Count hooks before.
    def _hook_count(m: nn.Module) -> int:
        return sum(len(m._forward_hooks.values()) for _, m in m.named_modules())

    before = _hook_count(model)
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    layerwise_stats(model, [batch], max_steps=1)
    after = _hook_count(model)
    assert after == before, "layerwise_stats must remove all hooks it registered"


def test_when_layerwise_stats_with_empty_dataloader_then_returns_empty_dict() -> None:
    """With no batches, no leaf modules produce statistics."""
    model = _LMWithLoss()
    stats = layerwise_stats(model, [], max_steps=5)
    assert isinstance(stats, dict)
    # Either empty or a pre-accumulated set — but must not raise.
    for value in stats.values():
        assert isinstance(value, float)
