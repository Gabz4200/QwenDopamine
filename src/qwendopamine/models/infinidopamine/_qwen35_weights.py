"""Qwen3.5 weight loading for InfiniDopamine.

Extracted from :mod:`model` for size. The loader splits a Qwen3.5 state dict
into vision, text, and lm-head partitions, then applies each via the
appropriate submodule.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

_logger = logging.getLogger(__name__)


def load_qwen35_weights(
    model: Any,
    weights: dict[str, torch.Tensor] | nn.Module,
    strict: bool = True,
) -> _IncompatibleKeys:
    r"""load_qwen35_weights(model, weights, strict=True) -> _IncompatibleKeys

    Load Qwen3.5 weights into a multimodal model, splitting into vision,
    text, and LM-head state.

    ``mtp.*`` keys (multi-token prediction heads) are reported at
    ``WARNING`` level and dropped, instead of being silently
    ``continue``d (review M7). The previously-built ``load_info`` list
    is removed — review N1 — and replaced with structured logging at
    INFO level per partition when counts are non-zero.

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
    mtp_keys: list[str] = []

    for k, v in state_dict.items():
        # MoE and mtp filters come FIRST so a MoE key like
        # ``model.language_model.layers.0.mlp.gate.weight`` is dropped
        # before the language-model prefix branch adds it to text_state.
        if k.startswith("mtp."):
            mtp_keys.append(k)
            continue
        if any(
            k.endswith(suffix)
            for suffix in (
                ".mlp.gate.weight",
                ".mlp.experts.gate_up_proj",
                ".mlp.experts.down_proj",
                ".mlp.experts.up_proj",
                ".mlp.experts.gate_proj",
                ".mlp.shared_expert.gate_proj.weight",
                ".mlp.shared_expert.up_proj.weight",
                ".mlp.shared_expert.down_proj.weight",
                ".mlp.shared_expert_gate.weight",
            )
        ):
            # Qwen3.5-0.8B+ uses MoE; InfiniDopamine is non-MoE (review N6).
            mtp_keys.append(k)
            continue
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
        elif strict:
            text_state[k] = v

    if mtp_keys:
        _logger.warning(
            "Dropping %d unsupported keys (mtp.* or MoE submodules; "
            "InfiniDopamine is non-MoE per review N6): %s",
            len(mtp_keys),
            mtp_keys[:5] + (["..."] if len(mtp_keys) > 5 else []),
        )

    all_missing: list[str] = []
    all_unexpected: list[str] = []

    if vision_state:
        missing_v, unexpected_v = model.model.visual.load_state_dict(
            vision_state, strict=strict
        )
        all_missing.extend(missing_v)
        all_unexpected.extend(unexpected_v)
        _logger.info(
            "vision partition: loaded %d / %d keys (missing=%d unexpected=%d)",
            len(vision_state) - len(missing_v),
            len(vision_state),
            len(missing_v),
            len(unexpected_v),
        )

    if text_state:
        missing_t, unexpected_t = model.model.language_model.load_qwen35_weights(
            text_state, strict=strict
        )
        all_missing.extend(missing_t)
        all_unexpected.extend(unexpected_t)
        _logger.info(
            "text partition: loaded %d / %d keys (missing=%d unexpected=%d)",
            len(text_state) - len(missing_t),
            len(text_state),
            len(missing_t),
            len(unexpected_t),
        )

    if lm_head_state:
        model.lm_head.weight.data.copy_(lm_head_state["lm_head.weight"])
        _logger.info("lm_head: loaded 1 key")

    return _IncompatibleKeys(all_missing, all_unexpected)


__all__ = ["load_qwen35_weights"]
