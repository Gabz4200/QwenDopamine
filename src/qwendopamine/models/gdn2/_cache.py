"""Cache helpers for GDN-2.

Extracted from :mod:`block` for modularity. The helpers are stateless functions
parameterised by ``layer_idx`` (the cache slot index) so the wrapping
:class:`~qwendopamine.models.gdn2.block.GatedDeltaNet2` can delegate to them.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from transformers.cache_utils import Cache

try:
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
except ImportError:
    LinearAttentionCacheLayerMixin = type(None)  # type: ignore[misc, assignment]


def get_cache(
    layer_idx: int | None,
    past_key_values: Cache | dict[str, Any] | None,
) -> tuple[
    torch.Tensor | None,
    tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
]:
    r"""get_cache(layer_idx: int | None, past_key_values: Cache | dict[str, Any] | None) -> tuple[torch.Tensor | None, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None]

    Extract the recurrent state and short-conv state from a cache.

    Args:
        layer_idx (int | None): Layer index used for cache disambiguation.
        past_key_values (Cache | dict[str, Any] | None): The cache to read.

    Returns:
        tuple[torch.Tensor | None, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None]:
        ``(recurrent_state, conv_state)``.
    """
    if past_key_values is None:
        return None, None

    if isinstance(past_key_values, Cache):
        layers = getattr(past_key_values, "layers", [])
        if layer_idx is not None and layer_idx < len(layers):
            layer_cache = layers[layer_idx]

            rec_states = getattr(layer_cache, "recurrent_states", None)
            if rec_states is None:
                rec_state = getattr(layer_cache, "recurrent_state", None)
            elif isinstance(rec_states, torch.Tensor):
                rec_state = rec_states
            elif isinstance(rec_states, dict):
                rec_state = rec_states.get(0)
            elif isinstance(rec_states, (list, tuple)) and len(rec_states) > 0:
                rec_state = rec_states[0]
            else:
                rec_state = None

            conv_states = getattr(layer_cache, "conv_states", None)
            if conv_states is None:
                conv_state = getattr(layer_cache, "conv_state", None)
            elif isinstance(conv_states, dict):
                conv_state = (
                    conv_states.get(0),
                    conv_states.get(1),
                    conv_states.get(2),
                )
            elif isinstance(conv_states, (list, tuple)) and len(conv_states) == 3:
                conv_state = (conv_states[0], conv_states[1], conv_states[2])
            else:
                conv_state = None

            return rec_state, conv_state
        return None, None

    if isinstance(past_key_values, dict):
        rec = past_key_values.get("recurrent_state")
        conv = past_key_values.get("conv_state")
        return rec, conv

    return None, None


def update_cache(
    layer_idx: int | None,
    past_key_values: Cache | dict[str, Any] | None,
    recurrent_state: torch.Tensor | None,
    conv_state: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
    | None,
) -> None:
    r"""update_cache(layer_idx: int | None, past_key_values: Cache | dict[str, Any] | None, recurrent_state: torch.Tensor | None, conv_state: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None) -> None

    Persist the latest recurrent and short-conv states back into a cache.

    Args:
        layer_idx (int | None): Layer index used for cache disambiguation.
        past_key_values (Cache | dict[str, Any] | None): The cache to update in-place.
        recurrent_state (torch.Tensor | None): Updated recurrent state.
        conv_state (tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None):
            Updated short-conv state.
    """
    if past_key_values is None:
        return

    if layer_idx is not None and isinstance(past_key_values, Cache):
        layers = getattr(past_key_values, "layers", [])
        if layer_idx < len(layers):
            layer_cache = layers[layer_idx]
            is_recurrent_layer = (
                isinstance(layer_cache, LinearAttentionCacheLayerMixin)
                or hasattr(layer_cache, "update_recurrent_state")
                or hasattr(layer_cache, "recurrent_states")
            )
            if (
                is_recurrent_layer
                and hasattr(past_key_values, "update_recurrent_state")
                and recurrent_state is not None
            ):
                try:
                    past_key_values.update_recurrent_state(recurrent_state, layer_idx)
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    RuntimeError,
                    IndexError,
                ) as e:
                    from qwendopamine.models.gdn2.backend import _warn_fallback_once

                    _warn_fallback_once(f"update_recurrent_state failed: {e}")
            elif recurrent_state is not None:
                rec_dict = getattr(layer_cache, "recurrent_states", None)
                if isinstance(rec_dict, dict):
                    rec_dict[0] = recurrent_state
                elif hasattr(layer_cache, "recurrent_state"):
                    layer_cache.recurrent_state = recurrent_state

            if (
                is_recurrent_layer
                and hasattr(past_key_values, "update_conv_state")
                and conv_state is not None
            ):
                try:
                    past_key_values.update_conv_state(cast(Any, conv_state), layer_idx)
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    RuntimeError,
                    IndexError,
                ) as e:
                    from qwendopamine.models.gdn2.backend import _warn_fallback_once

                    _warn_fallback_once(f"update_conv_state failed: {e}")
            elif conv_state is not None:
                conv_dict = getattr(layer_cache, "conv_states", None)
                if isinstance(conv_dict, dict):
                    conv_dict[0] = conv_state[0]
                    conv_dict[1] = conv_state[1]
                    conv_dict[2] = conv_state[2]
                elif hasattr(layer_cache, "conv_state"):
                    layer_cache.conv_state = conv_state

    elif isinstance(past_key_values, dict):
        if recurrent_state is not None:
            past_key_values["recurrent_state"] = recurrent_state
        if conv_state is not None:
            past_key_values["conv_state"] = conv_state


__all__ = ["get_cache", "update_cache"]
