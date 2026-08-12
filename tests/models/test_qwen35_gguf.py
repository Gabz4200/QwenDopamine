import pytest

import torch

from qwendopamine.integrations.gguf import load_qwen35_from_gguf
from qwendopamine.integrations.tokenizer import load_qwen35_tokenizer


@pytest.mark.slow
def test_qwen35_gguf_loading_forward_pass():
    model = load_qwen35_from_gguf(
        "unsloth/Qwen3.5-0.8B-MTP-GGUF",
        device_map="cpu",
    )
    tokenizer = load_qwen35_tokenizer("unsloth/Qwen3.5-0.8B-MTP-GGUF")

    inp = tokenizer("Hello world", return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)

    assert out.logits.shape == (1, 2, 248320)


def test_qwen35_tokenizer_fallback():
    tokenizer = load_qwen35_tokenizer("unsloth/Qwen3.5-0.8B-MTP-GGUF")
    assert tokenizer is not None
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")
