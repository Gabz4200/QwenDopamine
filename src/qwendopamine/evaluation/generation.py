r"""Autoregressive text generation utilities."""

from __future__ import annotations

from typing import Any

import torch

from qwendopamine.utils import get_model_device


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    include_prompt: bool = True,
    **kwargs: Any,
) -> str:
    r"""Generate text continuation from a prompt."""
    model.eval()
    device = get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    out_tokens = outputs[0] if include_prompt else outputs[0][input_ids.shape[1] :]
    return tokenizer.decode(out_tokens, skip_special_tokens=True)


__all__ = ["generate_text"]
