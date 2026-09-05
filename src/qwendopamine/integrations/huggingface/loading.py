"""Loading helpers for QwenDopamine HF models and tokenizers.

Provides ``load_config`` and ``load_model`` / ``load_tokenizer`` entry points
that delegate to ``transformers`` ``Auto`` classes.
"""

from __future__ import annotations

from typing import Any

from qwendopamine.integrations.huggingface.configs import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def load_config(model_name: str, **kwargs: Any) -> PreTrainedConfig:
    r"""Load a Hugging Face model config.

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.load_config` static method body.

    Args:
        model_name (str): repo id or local path.
        **kwargs: extra keyword arguments forwarded to
            :meth:`transformers.AutoConfig.from_pretrained`.

    Returns:
        PretrainedConfig: loaded config.

    Raises:
        RuntimeError: if transformers is not installed.
    """
    if AutoConfig is None:
        raise RuntimeError(
            "transformers is required for HF config loading. Install qwendopamine[hf]."
        )
    return AutoConfig.from_pretrained(model_name, **kwargs)


def load_model(
    model_name: str,
    quantization_config: Any = None,
    device_map: str = "cpu",
    from_gguf: bool = False,
    **kwargs: Any,
) -> PreTrainedModel:
    r"""Load a Hugging Face causal-LM model, optionally with quantization.

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.load_model` static method body.

    If ``from_gguf`` is ``True`` or ``model_name`` ends with ``.gguf``,
    the loader uses ``gguf_file`` to load GGUF weights directly.

    Args:
        model_name (str): repo id, local path, or GGUF file path.
        quantization_config (Any, optional): quantization config. When
            ``None``, :func:`make_quantization_config` creates an ``int8``
            config with CPU offload automatically.
        device_map (str): device placement map. Default: ``"cpu"``.
        from_gguf (bool): force GGUF loading. Default: ``False``.
        **kwargs: extra keyword arguments forwarded to
            :meth:`transformers.AutoModelForCausalLM.from_pretrained`.

    Returns:
        PreTrainedModel: loaded model.

    Raises:
        RuntimeError: if transformers is not installed.
    """
    if AutoModelForCausalLM is None:
        raise RuntimeError(
            "transformers is required for HF model loading. Install qwendopamine[hf]."
        )

    if from_gguf or model_name.endswith(".gguf"):
        gguf_result: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore[assignment]
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            gguf_file=model_name if model_name.endswith(".gguf") else None,
            **kwargs,
        )
        return gguf_result

    result2: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore[assignment]
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        **kwargs,
    )
    return result2


def load_tokenizer(
    model_name: str, **kwargs: Any
) -> PreTrainedTokenizer | PreTrainedTokenizerFast | None:
    r"""Load a Hugging Face tokenizer.

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.load_tokenizer` static method body.

    Args:
        model_name (str): repo id or local path.
        **kwargs: extra keyword arguments forwarded to
            :meth:`transformers.AutoTokenizer.from_pretrained`.

    Returns:
        PreTrainedTokenizer | PreTrainedTokenizerFast | None: loaded tokenizer.

    Raises:
        RuntimeError: if transformers is not installed.
    """
    if AutoTokenizer is None:
        raise RuntimeError(
            "transformers is required for tokenizer loading. Install qwendopamine[hf]."
        )
    return AutoTokenizer.from_pretrained(model_name, **kwargs)
