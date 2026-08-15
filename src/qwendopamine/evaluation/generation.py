from __future__ import annotations

from typing import Any

import torch

from qwendopamine.utils import get_model_device as _get_model_device


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    **kwargs: Any,
) -> str:
    r"""Generate text from a prompt using the model's ``generate`` method.

    Args:
        model (torch.nn.Module): causal language model with ``generate``.
        tokenizer (Any): tokenizer with ``__call__`` and ``decode``.
        prompt (str): text prompt to continue.
        max_new_tokens (int): maximum number of tokens to generate. Default: ``256``.
        **kwargs: additional keyword arguments forwarded to ``model.generate``.

    Returns:
        str: decoded generated text without special tokens.
    """
    model.eval()
    device = _get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )  # type: ignore[arg-type]

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
