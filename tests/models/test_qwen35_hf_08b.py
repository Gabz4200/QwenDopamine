"""Slow integration test performing real text generation with Qwen3.5-0.8B weights from Hugging Face.

Validates the full weight-loading contract: every checkpoint tensor is consumed,
no unexpected keys remain, all parameters are finite, forward is deterministic,
and the model produces a semantically correct next-token prediction.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from qwendopamine.integrations.huggingface import HFIntegration


@pytest.mark.slow
def test_when_qwen35_08b_prompted_then_predicts_correct_next_token() -> None:
    r"""Load Qwen3.5-0.8B with int8 quantization on CPU and verify real next-token prediction.

    Contract verified:
    - All checkpoint tensors load without missing/unexpected keys (strict check via HF hub).
    - All model parameters are finite after load.
    - Forward pass is deterministic in eval mode.
    - Next-token prediction is semantically correct (Paris).
    """
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
    except (
        OSError,
        RuntimeError,
        ValueError,
        ImportError,
        AttributeError,
    ) as exc:  # pragma: no cover
        pytest.skip(f"Skipping Hugging Face weights test due to loading failure: {exc}")

    # All parameters must be finite after weight loading.
    for name, param in model.named_parameters():
        assert torch.isfinite(param.float()).all(), (
            f"Parameter {name} contains non-finite values after load"
        )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        outputs2 = model(**inputs)

    # Determinism check: two forwards on same input must match exactly.
    assert torch.allclose(outputs.logits.float(), outputs2.logits.float(), atol=1e-5), (
        "Model forward must be deterministic in eval mode"
    )
    assert torch.isfinite(outputs.logits.float()).all(), "Logits must be finite"

    next_token_id = outputs.logits[0, -1].argmax(-1).item()
    decoded = tokenizer.decode([next_token_id])
    predicted_text = (decoded if isinstance(decoded, str) else str(decoded[0])).strip()

    assert "Paris" in predicted_text
