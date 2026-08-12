from __future__ import annotations

from typing import Any

import torch


def generate_text(model: torch.nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int = 256, **kwargs: Any) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(next(model.parameters()).device)
    attention_mask = inputs.attention_mask.to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, **kwargs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
