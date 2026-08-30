"""Shared Hugging Face compatibility fallbacks for QwenDopamine model ports.

This module centralizes the optional-import fallbacks and compatibility shims
that were previously duplicated across ``modular_infinidopamine.py`` and
``modular_qwen3_5.py``. Importing from a single source prevents silent drift
when the ``transformers`` API changes.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

try:
    from huggingface_hub.dataclasses import strict as _hf_strict

    def strict(cls: Any) -> Any:
        try:
            return _hf_strict(cls)
        except (TypeError, ValueError):
            return cls
except ImportError:

    def strict(cls: Any) -> Any:
        return cls


try:
    from transformers import initialization as init
except ImportError:

    class _InitFallback:
        @staticmethod
        def ones_(tensor: nn.Parameter) -> Any:
            return nn.init.ones_(tensor)

        @staticmethod
        def copy_(target: nn.Parameter, source: nn.Parameter) -> Any:
            return target.copy_(source)

    init = _InitFallback()  # type: ignore[assignment]


try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:

    class Cache:  # type: ignore[no-redef]
        pass

    class DynamicCache(Cache):  # type: ignore[no-redef]
        pass


try:
    from transformers.integrations import (
        use_kernel_forward_from_hub as _hf_use_kernel_forward_from_hub,
    )
    from transformers.integrations import (
        use_kernelized_func as _hf_use_kernelized_func,
    )

    def use_kernel_forward_from_hub(*args: Any, **kwargs: Any) -> Any:
        try:
            inner = _hf_use_kernel_forward_from_hub(*args, **kwargs)
        except (ImportError, RuntimeError, TypeError, ValueError):

            def noop_decorator(fn: Any) -> Any:
                return fn

            return noop_decorator

        def decorator(fn: Any) -> Any:
            try:
                return inner(fn)
            except (RuntimeError, TypeError, ValueError):
                return fn

        return decorator

    def use_kernelized_func(*args: Any, **kwargs: Any) -> Any:
        try:
            inner = _hf_use_kernelized_func(*args, **kwargs)
        except (ImportError, RuntimeError, TypeError, ValueError):

            def noop_decorator(fn: Any) -> Any:
                return fn

            return noop_decorator

        def decorator(fn: Any) -> Any:
            try:
                return inner(fn)
            except (RuntimeError, TypeError, ValueError):
                return fn

        return decorator
except ImportError:

    def use_kernel_forward_from_hub(*args: Any, **kwargs: Any) -> Any:
        def decorator(fn: Any) -> Any:
            return fn

        return decorator

    def use_kernelized_func(*args: Any, **kwargs: Any) -> Any:
        def decorator(fn: Any) -> Any:
            return fn

        return decorator


try:
    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    try:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
    except ImportError:

        def create_causal_mask(*args: Any, **kwargs: Any) -> Any:
            return None

    def create_recurrent_attention_mask(*args: Any, **kwargs: Any) -> Any:
        return None

    def create_sliding_window_causal_mask(*args: Any, **kwargs: Any) -> Any:
        return None


try:
    from transformers.modeling_layers import (
        GenericForSequenceClassification,
        GenericForTokenClassification,
        GradientCheckpointingLayer,
    )
except ImportError:

    class GenericForSequenceClassification:  # type: ignore[no-redef]
        pass

    class GenericForTokenClassification:  # type: ignore[no-redef]
        pass

    class GradientCheckpointingLayer(nn.Module):  # type: ignore[no-redef]
        pass


try:
    from transformers.modeling_outputs import (
        BaseModelOutputWithPast,
        BaseModelOutputWithPooling,
        CausalLMOutputWithPast,
        SequenceClassifierOutputWithPast,
    )
except ImportError:

    class BaseModelOutputWithPast:  # type: ignore[no-redef]
        pass

    class BaseModelOutputWithPooling:  # type: ignore[no-redef]
        pass

    class CausalLMOutputWithPast:  # type: ignore[no-redef]
        pass

    class SequenceClassifierOutputWithPast:  # type: ignore[no-redef]
        pass


try:
    from transformers.modeling_utils import PreTrainedModel
except ImportError:

    class PreTrainedModel(nn.Module):  # type: ignore[no-redef]
        pass


try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
except (ImportError, ModuleNotFoundError, AttributeError):

    class Qwen3ForCausalLM(nn.Module):  # type: ignore[no-redef]
        pass


try:
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
except (ImportError, ModuleNotFoundError, AttributeError):

    class Qwen3NextConfig:  # type: ignore[no-redef]
        pass


try:
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextAttention,
        Qwen3NextGatedDeltaNet,
        Qwen3NextMLP,
        Qwen3NextModel,
        Qwen3NextPreTrainedModel,
        Qwen3NextRMSNorm,
        Qwen3NextSparseMoeBlock,
        apply_mask_to_padding_states,
        causal_conv1d_fn,
        causal_conv1d_update,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )
except (ImportError, ModuleNotFoundError, AttributeError):

    class Qwen3NextAttention(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextGatedDeltaNet(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextMLP(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextModel(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextPreTrainedModel(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextRMSNorm(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3NextSparseMoeBlock(nn.Module):  # type: ignore[no-redef]
        pass

    apply_mask_to_padding_states = None  # type: ignore[misc, assignment]
    causal_conv1d_fn = None  # type: ignore[misc, assignment]
    causal_conv1d_update = None  # type: ignore[misc, assignment]
    torch_chunk_gated_delta_rule = None  # type: ignore[misc, assignment]
    torch_recurrent_gated_delta_rule = None  # type: ignore[misc, assignment]


try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLVisionConfig,
    )
except (ImportError, ModuleNotFoundError, AttributeError):

    class Qwen3VLConfig:  # type: ignore[no-redef]
        pass

    class Qwen3VLVisionConfig:  # type: ignore[no-redef]
        pass


try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
        Qwen3VLModelOutputWithPast,
        Qwen3VLTextRotaryEmbedding,
        Qwen3VLVisionModel,
        Qwen3VLVisionRotaryEmbedding,
    )
except (ImportError, ModuleNotFoundError, AttributeError):

    class Qwen3VLForConditionalGeneration(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3VLModel(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3VLModelOutputWithPast:  # type: ignore[no-redef]
        pass

    class Qwen3VLTextRotaryEmbedding(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3VLVisionModel(nn.Module):  # type: ignore[no-redef]
        pass

    class Qwen3VLVisionRotaryEmbedding(nn.Module):  # type: ignore[no-redef]
        pass


try:
    from typing import Unpack
except ImportError:
    try:
        from transformers.processing_utils import Unpack
    except ImportError:
        Unpack = Any  # type: ignore[misc, assignment]


try:
    from transformers.utils import (
        TransformersKwargs,
        can_return_tuple,
        logging,
    )
except ImportError:

    def can_return_tuple(fn: Any) -> Any:
        return fn

    TransformersKwargs = Any  # type: ignore[misc, assignment]
    logging = logging  # type: ignore[assignment]  # noqa: PLW0127


try:
    from transformers.utils.generic import (
        accepts_precomputed_kwargs,
        merge_with_config_defaults,
    )
except ImportError:

    def accepts_precomputed_kwargs(*args: Any, **kwargs: Any) -> Any:
        def decorator(fn: Any) -> Any:
            return fn

        return decorator

    def merge_with_config_defaults(fn: Any) -> Any:
        return fn


try:
    from transformers.utils.output_capturing import capture_outputs
except ImportError:

    def capture_outputs(fn: Any) -> Any:
        return fn


try:
    from transformers.vision_utils import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )
except ImportError:
    get_vision_attention_seqlens = None  # type: ignore[misc, assignment]
    get_vision_interpolation_indices_and_weights = None  # type: ignore[misc, assignment]
    get_vision_position_ids = None  # type: ignore[misc, assignment]


logger = logging.get_logger(__name__)  # type: ignore[attr-defined]


def expand_position_ids_to_multimodal(
    position_ids: torch.LongTensor | None,
    batch_size: int,
    seq_len: int,
    past_seen_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Expand 1D/2D position ids into the 4D multimodal layout.

    When ``position_ids`` is ``None`` a fresh 1D ``torch.arange`` is created
    and expanded. Explicit 2D inputs are expanded in-place; all other shapes
    pass through unchanged.

    Returns the expanded position ids and the extracted ``text_position_ids``
    (the first slice), or ``None`` when the input does not match the expected
    4D shape.
    """
    if position_ids is None:
        expanded = torch.arange(seq_len, device=device) + past_seen_tokens
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
    """Unwrap ``__wrapped__`` decorators on qwen3_next gated-delta-rule functions.

    Some ``transformers`` builds wrap these functions with decorators that are
    incompatible with CPU execution or custom autograd. This helper removes the
    wrappers in-place so callers can use the raw implementations.
    """
    import torch as _torch

    if _torch.cuda.is_available():
        return
    global torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule
    global causal_conv1d_fn, causal_conv1d_update
    if (
        torch_chunk_gated_delta_rule is not None
        and hasattr(torch_chunk_gated_delta_rule, "__wrapped__")
    ):
        while hasattr(torch_chunk_gated_delta_rule, "__wrapped__"):
            torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule.__wrapped__
    if (
        torch_recurrent_gated_delta_rule is not None
        and hasattr(torch_recurrent_gated_delta_rule, "__wrapped__")
    ):
        while hasattr(torch_recurrent_gated_delta_rule, "__wrapped__"):
            torch_recurrent_gated_delta_rule = (
                torch_recurrent_gated_delta_rule.__wrapped__
            )
    if causal_conv1d_fn is not None and hasattr(causal_conv1d_fn, "__wrapped__"):
        while hasattr(causal_conv1d_fn, "__wrapped__"):
            causal_conv1d_fn = causal_conv1d_fn.__wrapped__
    if causal_conv1d_update is not None and hasattr(causal_conv1d_update, "__wrapped__"):
        while hasattr(causal_conv1d_update, "__wrapped__"):
            causal_conv1d_update = causal_conv1d_update.__wrapped__
    try:
        import transformers.models.qwen3_next.modeling_qwen3_next as _q3n

        for _name in [
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
            "causal_conv1d_update",
        ]:
            if hasattr(_q3n, _name):
                _fn = getattr(_q3n, _name)
                while hasattr(_fn, "__wrapped__"):
                    _fn = _fn.__wrapped__
                setattr(_q3n, _name, _fn)
    except (ImportError, AttributeError):
        pass


__all__ = [
    "BaseModelOutputWithPast",
    "BaseModelOutputWithPooling",
    "CausalLMOutputWithPast",
    "DynamicCache",
    "GenericForSequenceClassification",
    "GenericForTokenClassification",
    "GradientCheckpointingLayer",
    "PreTrainedModel",
    "Qwen3ForCausalLM",
    "Qwen3NextAttention",
    "Qwen3NextConfig",
    "Qwen3NextGatedDeltaNet",
    "Qwen3NextMLP",
    "Qwen3NextModel",
    "Qwen3NextPreTrainedModel",
    "Qwen3NextRMSNorm",
    "Qwen3NextSparseMoeBlock",
    "Qwen3VLConfig",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLModelOutputWithPast",
    "Qwen3VLTextRotaryEmbedding",
    "Qwen3VLVisionConfig",
    "Qwen3VLVisionModel",
    "Qwen3VLVisionRotaryEmbedding",
    "SequenceClassifierOutputWithPast",
    "TransformersKwargs",
    "Unpack",
    "accepts_precomputed_kwargs",
    "apply_mask_to_padding_states",
    "can_return_tuple",
    "capture_outputs",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "create_causal_mask",
    "create_recurrent_attention_mask",
    "create_sliding_window_causal_mask",
    "expand_position_ids_to_multimodal",
    "get_vision_attention_seqlens",
    "get_vision_interpolation_indices_and_weights",
    "get_vision_position_ids",
    "init",
    "logger",
    "logging",
    "merge_with_config_defaults",
    "strict",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "unwrap_gated_delta_rule_fns",
    "use_kernel_forward_from_hub",
    "use_kernelized_func",
]
