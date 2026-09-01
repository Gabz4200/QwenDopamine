"""Real utility functions used by the Qwen3.5 and InfiniDopamine HF ports.

This module contains only side-effecting logic that depends on ``transformers``.
Symbol re-exports of ``transformers`` itself happen at the call site, not here.
"""

from __future__ import annotations

import torch as _torch


def expand_position_ids_to_multimodal(
    position_ids: _torch.LongTensor | None,
    batch_size: int,
    seq_len: int,
    past_seen_tokens: int,
    device: _torch.device,
) -> tuple[_torch.Tensor, _torch.Tensor | None]:
    r"""Expand 1D/2D position ids into the 4D multimodal layout.

    When ``position_ids`` is ``None`` a fresh 1D ``torch.arange`` is created
    and expanded. Explicit 2D inputs are expanded in-place; all other shapes
    pass through unchanged.

    Returns the expanded position ids and the extracted ``text_position_ids``
    (the first slice), or ``None`` when the input does not match the expected
    4D shape.
    """
    if position_ids is None:
        expanded = _torch.arange(seq_len, device=device) + past_seen_tokens
        expanded = expanded.view(1, 1, -1).expand(4, batch_size, -1)
    elif position_ids.ndim == 2:
        expanded = position_ids[:, None, :].expand(4, batch_size, -1)
    else:
        expanded = position_ids

    if expanded.ndim == 3 and expanded.shape[0] == 4:
        text_position_ids = expanded[0]
        expanded = expanded[1:]
    else:
        text_position_ids = None
    return expanded, text_position_ids


def unwrap_gated_delta_rule_fns() -> None:
    r"""Unwrap ``__wrapped__`` decorators on qwen3_next gated-delta-rule functions.

    Some ``transformers`` builds wrap these functions with decorators that are
    incompatible with CPU execution or custom autograd. This helper removes the
    wrappers in-place so callers can use the raw implementations.
    """
    import torch as _torch

    if _torch.cuda.is_available():
        return

    import transformers.models.qwen3_next.modeling_qwen3_next as _q3n
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        causal_conv1d_fn as _causal_conv1d_fn,
    )
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        causal_conv1d_update as _causal_conv1d_update,
    )
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_chunk_gated_delta_rule as _torch_chunk_gated_delta_rule,
    )
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_recurrent_gated_delta_rule as _torch_recurrent_gated_delta_rule,
    )

    for _name, _fn in [
        ("torch_chunk_gated_delta_rule", _torch_chunk_gated_delta_rule),
        ("torch_recurrent_gated_delta_rule", _torch_recurrent_gated_delta_rule),
        ("causal_conv1d_fn", _causal_conv1d_fn),
        ("causal_conv1d_update", _causal_conv1d_update),
    ]:
        if _fn is None:
            continue
        while hasattr(_fn, "__wrapped__"):
            _fn = _fn.__wrapped__
        if hasattr(_q3n, _name):
            setattr(_q3n, _name, _fn)


__all__ = [
    "expand_position_ids_to_multimodal",
    "unwrap_gated_delta_rule_fns",
]