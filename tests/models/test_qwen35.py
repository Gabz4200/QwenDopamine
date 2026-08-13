r"""Tests for Qwen3.5 reference model forward pass and generation.

Both tests use 8-bit quantization so they run on CPU in reasonable time.
"""
from __future__ import annotations

import pytest
import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)


@pytest.mark.slow
def test_qwen35_forward_pass_shape():
    r"""Qwen3.5 model loaded from pretrained (8-bit quantized) must output expected logits shape."""
    model_name = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = Qwen3_5TextConfig.from_pretrained(model_name)
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    model = Qwen3_5ForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="cpu")

    inp = tokenizer("Hello world", return_tensors="pt")  # type: ignore[operator]
    with torch.no_grad():
        out = model(**inp)

    assert out.logits.shape == (1, 2, config.vocab_size)


@pytest.mark.slow
def test_qwen35_generation_smoke():
    r"""Pretrained Qwen3.5 model (8-bit quantized) must generate text starting with the prompt."""
    model_name = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    model = Qwen3_5ForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="cpu", torch_dtype="auto")

    inp = tokenizer("Hello", return_tensors="pt")  # type: ignore[operator]
    out = model.generate(**inp, max_new_tokens=10)  # type: ignore[arg-type]
    text = tokenizer.decode(out[0])  # type: ignore[union-attr]
    assert text.startswith("Hello")
