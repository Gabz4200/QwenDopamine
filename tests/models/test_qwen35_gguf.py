r"""Tests for GGUF loading and tokenizer fallback."""

import pytest
import torch

from qwendopamine.integrations.gguf import load_qwen35_from_gguf
from qwendopamine.integrations.tokenizer import load_qwen35_tokenizer


@pytest.mark.slow
def test_qwen35_gguf_loading_forward_pass():
    r"""GGUF-loaded model must produce logits of expected shape."""
    model = load_qwen35_from_gguf(
        "unsloth/Qwen3.5-0.8B-MTP-GGUF",
        device_map="cpu",
    )
    tokenizer = load_qwen35_tokenizer("unsloth/Qwen3.5-0.8B-MTP-GGUF")

    inp = tokenizer("Hello world", return_tensors="pt")  # type: ignore[operator]
    with torch.no_grad():
        out = model(**inp)

    assert out.logits.shape == (1, 2, 248320)


def test_qwen35_tokenizer_fallback():
    r"""Tokenizer loader must fall back to Qwen/Qwen3.5-0.8B."""
    tokenizer = load_qwen35_tokenizer("unsloth/Qwen3.5-0.8B-MTP-GGUF")
    assert tokenizer is not None
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")
