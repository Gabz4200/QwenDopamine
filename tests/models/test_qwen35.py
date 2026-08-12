from __future__ import annotations

import torch

from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig


def test_qwen35_forward_pass_shape():
    model_name = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = Qwen3_5TextConfig.from_pretrained(model_name)
    model = Qwen3_5ForCausalLM(config=config)

    inp = tokenizer("Hello world", return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)

    assert out.logits.shape == (1, 2, config.vocab_size)


def test_qwen35_generation_smoke():
    model_name = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen3_5ForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto")

    inp = tokenizer("Hello", return_tensors="pt")
    out = model.generate(**inp, max_new_tokens=10)
    text = tokenizer.decode(out[0])
    assert text.startswith("Hello")
