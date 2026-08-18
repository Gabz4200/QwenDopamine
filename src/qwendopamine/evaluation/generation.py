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
    **kwargs: Any,
) -> str:
    r"""generate_text(model, tokenizer, prompt, max_new_tokens=256, **kwargs) -> str

    Generates text continuation from a prompt using model generation method and tokenizer decoding.

    Args:
        model (Any): Causal language model instance with ``generate`` method.
        tokenizer (Any): Tokenizer instance with encoding and decoding methods.
        prompt (str): Text prompt string to continue.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Default: ``256``.
        **kwargs (Any): Additional keyword arguments passed to ``model.generate``.

    Returns:
        str: Decoded generated output text string without special tokens.
    """
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

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


__all__ = ["generate_text"]
