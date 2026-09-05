"""Qwen3.5 weight loading for InfiniDopamine.

Extracted from :mod:`model` for size. The loader splits a Qwen3.5 state dict
into vision, text, and lm-head partitions, then applies each via the
appropriate submodule.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys


def load_qwen35_weights(
    model: Any,
    weights: dict[str, torch.Tensor] | nn.Module,
    strict: bool = True,
) -> _IncompatibleKeys:
    r"""load_qwen35_weights(model, weights, strict=True) -> _IncompatibleKeys

    Load Qwen3.5 weights into a multimodal model, splitting into vision,
    text, and LM-head state.

    Args:
        model (Any): The :class:`InfiniDopamineForConditionalGeneration` instance.
        weights (dict[str, torch.Tensor] | nn.Module): State dict or module.
        strict (bool): Strict load. Default: ``True``.

    Returns:
        _IncompatibleKeys: Named tuple of ``missing_keys`` and
        ``unexpected_keys``.
    """
    if isinstance(weights, nn.Module):
        state_dict = weights.state_dict()
    else:
        state_dict = dict(weights)

    vision_state: dict[str, torch.Tensor] = {}
    text_state: dict[str, torch.Tensor] = {}
    lm_head_state: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        if k == "lm_head.weight":
            lm_head_state[k] = v
        elif k.startswith("model.visual."):
            vision_state[k[len("model.visual.") :]] = v
        elif k.startswith("model.language_model."):
            text_state[k[len("model.") :]] = v
        elif k.startswith("language_model."):
            text_state[k] = v
        elif k.startswith("visual."):
            vision_state[k[len("visual.") :]] = v
        elif k.startswith("mtp."):
            continue
        elif strict:
            text_state[k] = v

    all_missing: list[str] = []
    all_unexpected: list[str] = []
    load_info: list[str] = []

    if vision_state:
        missing_v, unexpected_v = model.model.visual.load_state_dict(
            vision_state, strict=strict
        )
        all_missing.extend(missing_v)
        all_unexpected.extend(unexpected_v)
        load_info.append(
            f"vision: loaded {len(vision_state) - len(missing_v)} keys "
            f"({len(missing_v)} missing, {len(unexpected_v)} unexpected)"
        )

    if text_state:
        missing_t, unexpected_t = model.model.language_model.load_qwen35_weights(
            text_state, strict=strict
        )
        all_missing.extend(missing_t)
        all_unexpected.extend(unexpected_t)
        load_info.append(
            f"text: loaded {len(text_state) - len(missing_t)} keys "
            f"({len(missing_t)} missing, {len(unexpected_t)} unexpected)"
        )

    if lm_head_state:
        model.lm_head.weight.data.copy_(lm_head_state["lm_head.weight"])
        load_info.append("lm_head: loaded 1 key")

    return _IncompatibleKeys(all_missing, all_unexpected)


__all__ = ["load_qwen35_weights"]
