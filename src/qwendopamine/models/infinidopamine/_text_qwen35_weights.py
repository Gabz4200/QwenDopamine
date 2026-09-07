"""Text-tower Qwen3.5 weight loading for InfiniDopamine.

Extracted from :mod:`model` for size. The loader strips visual/lm_head/MTP
prefixes from a Qwen3.5 state dict and applies the rest to the
:class:`InfiniDopamineTextModel`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

_logger = logging.getLogger(__name__)


def _is_dropped_unsupported(k: str) -> bool:
    """Return True if ``k`` is a weight InfiniDopamine does not support.

    Currently covers:
      - ``mtp.*`` (multi-token prediction heads) — review M7
      - MoE submodules (``mlp.gate``, ``mlp.experts.*``,
        ``mlp.shared_expert.*``) — Qwen3.5-0.8B is MoE; InfiniDopamine
        is non-MoE (review N6). The dense ``mlp.gate_proj``,
        ``mlp.up_proj``, ``mlp.down_proj`` are the supported path.
    """
    if k.startswith("mtp."):
        return True
    unsupported_suffixes = (
        ".mlp.gate.weight",
        ".mlp.experts.gate_up_proj",
        ".mlp.experts.up_proj",
        ".mlp.experts.down_proj",
        ".mlp.experts.gate_proj",
        ".mlp.shared_expert.gate_proj.weight",
        ".mlp.shared_expert.up_proj.weight",
        ".mlp.shared_expert.down_proj.weight",
        ".mlp.shared_expert_gate.weight",
    )
    return any(k.endswith(s) for s in unsupported_suffixes)


def _warn_mtp_drops(state_dict: dict[str, torch.Tensor]) -> None:
    """Emit a warning if any unsupported keys would be dropped (review M7)."""
    dropped = [k for k in state_dict if _is_dropped_unsupported(k)]
    if dropped:
        _logger.warning(
            "Dropping %d unsupported keys (mtp.* or MoE submodules; "
            "InfiniDopamine is non-MoE per review N6): %s",
            len(dropped),
            dropped[:5] + (["..."] if len(dropped) > 5 else []),
        )


def load_text_qwen35_weights(
    model: Any,
    weights: dict[str, torch.Tensor] | nn.Module,
    strict: bool = True,
) -> Any:
    r"""load_text_qwen35_weights(model, weights, strict=True) -> Any

    Load pretrained Qwen3.5 (GDN-1) weights into InfiniDopamine (GDN-2
    with SWA), stripping visual and MTP prefixes.

    Args:
        model (Any): The :class:`InfiniDopamineTextModel` instance.
        weights (dict[str, torch.Tensor] | nn.Module): State dict or module.
        strict (bool): Strict load. Default: ``True``.

    Returns:
        Any: Result of :meth:`load_state_dict` (missing/unexpected keys).
    """
    if isinstance(weights, nn.Module):
        state_dict = weights.state_dict()
    else:
        state_dict = dict(weights)

    has_full_prefix = any(
        k.startswith(("model.language_model.", "model.", "language_model."))
        for k in state_dict
    )
    if has_full_prefix:
        _warn_mtp_drops(state_dict)
        remapped_state_dict: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            # Skip MoE / mtp / visual / lm_head keys; these are
            # filtered regardless of prefix.
            if (
                _is_dropped_unsupported(k)
                or k.startswith(("model.visual.", "visual."))
                or k == "lm_head.weight"
            ):
                continue
            new_k = k
            if new_k.startswith("model.language_model."):
                new_k = new_k[len("model.language_model.") :]
            elif new_k.startswith("language_model."):
                new_k = new_k[len("language_model.") :]
            elif new_k.startswith("model."):
                new_k = new_k[len("model.") :]
            if not k.startswith(("model.visual.", "visual.", "mtp.")):
                remapped_state_dict[new_k] = v
        state_dict = remapped_state_dict

    return model.load_state_dict(state_dict, strict=strict)


def load_causal_lm_qwen35_weights(
    model: Any,
    weights: dict[str, torch.Tensor] | nn.Module,
    strict: bool = True,
) -> Any:
    r"""load_causal_lm_qwen35_weights(model, weights, strict=True) -> Any

    Load Qwen3.5 (GDN-1) weights into a text-only causal LM, remapping
    language-model prefixes.

    Args:
        model (Any): The :class:`InfiniDopamineForCausalLM` instance.
        weights (dict[str, torch.Tensor] | nn.Module): State dict or module.
        strict (bool): Strict load. Default: ``True``.

    Returns:
        Any: Result of :meth:`load_state_dict`.
    """
    if isinstance(weights, nn.Module):
        state_dict = weights.state_dict()
    else:
        state_dict = dict(weights)

    has_language_model_prefix = any(
        k.startswith(("model.language_model.", "language_model.")) for k in state_dict
    )
    if has_language_model_prefix:
        _warn_mtp_drops(state_dict)
        remapped_state_dict: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            # Skip MoE / mtp / visual keys; ``lm_head.weight`` is kept
            # because the causal_lm owns its own lm_head.
            if _is_dropped_unsupported(k) or k.startswith(("model.visual.", "visual.")):
                continue
            if k.startswith("model.language_model."):
                remapped_state_dict[k.replace("model.language_model.", "model.")] = v
            elif k.startswith("language_model."):
                remapped_state_dict[k.replace("language_model.", "model.")] = v
            elif k == "lm_head.weight" or not k.startswith(
                ("model.visual.", "visual.", "mtp.")
            ):
                remapped_state_dict[k] = v
        state_dict = remapped_state_dict

    return model.load_state_dict(state_dict, strict=strict)


__all__ = ["load_causal_lm_qwen35_weights", "load_text_qwen35_weights"]
