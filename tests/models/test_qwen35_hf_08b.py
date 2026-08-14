"""Slow integration test performing real text generation with Qwen3.5-0.8B weights from Hugging Face."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from qwendopamine.integrations.huggingface import HFIntegration


@pytest.mark.slow
def test_when_qwen35_08b_prompted_then_predicts_correct_next_token() -> None:
    r"""Load Qwen3.5-0.8B with int8 quantization on CPU and verify real next-token prediction."""
    model_name = "Qwen/Qwen3.5-0.8B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert tokenizer is not None
        qconfig = HFIntegration.make_quantization_config(method="int8", device="cpu")
        model = HFIntegration.load_model(
            model_name=model_name,
            quantization_config=qconfig,
            device_map="cpu",
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        )
    except (OSError, RuntimeError, ValueError, ImportError) as exc:  # pragma: no cover
        pytest.skip(f"Skipping Hugging Face weights test due to loading failure: {exc}")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    next_token_id = outputs.logits[0, -1].argmax(-1).item()
    decoded = tokenizer.decode([next_token_id])
    predicted_text = (decoded if isinstance(decoded, str) else str(decoded[0])).strip()

    assert "Paris" in predicted_text
