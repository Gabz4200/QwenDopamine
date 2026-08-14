from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        QuantoConfig,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    @dataclass
    class _FallbackBitsAndBytesConfig:
        load_in_8bit: bool = False
        llm_int8_enable_fp32_cpu_offload: bool = False
        load_in_4bit: bool = False
        bnb_4bit_quant_type: str = "nf4"
        bnb_4bit_compute_dtype: Any = None

    @dataclass
    class _FallbackQuantoConfig:
        weights: str = "int8"

    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = _FallbackBitsAndBytesConfig
    PreTrainedConfig = Any
    PreTrainedModel = Any
    PreTrainedTokenizer = Any
    PreTrainedTokenizerFast = Any
    QuantoConfig = _FallbackQuantoConfig


class HFIntegration:
    @staticmethod
    def make_quantization_config(
        method: str = "int8", compute_dtype: str = "bfloat16", device: str = "cpu"
    ) -> BitsAndBytesConfig | QuantoConfig:
        r"""Build a Hugging Face quantization config for CPU-friendly loading.

        Args:
            method (str): quantization method. Accepted values: ``"int8"``,
                ``"int4"``, or a BitsAndBytes/Quanto weight type.
            compute_dtype (str): compute dtype name passed to
                :class:`torch.dtype`. Default: ``"bfloat16"``.
            device (str): target device string. Used to enable CPU offload for
                ``int8``. Default: ``"cpu"``.

        Returns:
            BitsAndBytesConfig | QuantoConfig: quantization config object.
        """
        if method == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=device == "cpu"
            )
        if method == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, compute_dtype, torch.bfloat16),
            )
        return QuantoConfig(weights=method)

    @staticmethod
    def load_config(model_name: str, **kwargs: Any) -> PreTrainedConfig:
        r"""Load a Hugging Face model config.

        Args:
            model_name (str): repo id or local path.
            **kwargs: extra keyword arguments forwarded to
                :meth:`transformers.AutoConfig.from_pretrained`.

        Returns:
            PretrainedConfig: loaded config.
        """
        if AutoConfig is None:
            raise RuntimeError(
                "transformers is required for HF config loading. Install qwendopamine[hf]."
            )
        return AutoConfig.from_pretrained(model_name, **kwargs)

    @staticmethod
    def load_model(
        model_name: str,
        quantization_config: Any = None,
        device_map: str = "cpu",
        from_gguf: bool = False,
        **kwargs: Any,
    ) -> PreTrainedModel:
        r"""Load a Hugging Face causal-LM model, optionally with quantization.

        If ``from_gguf`` is ``True`` or ``model_name`` ends with ``.gguf``,
        the loader uses ``gguf_file`` to load GGUF weights directly.

        Args:
            model_name (str): repo id, local path, or GGUF file path.
            quantization_config (Any, optional): quantization config. When
                ``None``, :meth:`make_quantization_config` creates an ``int8``
                config with CPU offload automatically.
            device_map (str): device placement map. Default: ``"cpu"``.
            from_gguf (bool): force GGUF loading. Default: ``False``.
            **kwargs: extra keyword arguments forwarded to
                :meth:`transformers.AutoModelForCausalLM.from_pretrained`.

        Returns:
            PreTrainedModel: loaded model.
        """
        if AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers is required for HF model loading. Install qwendopamine[hf]."
            )
        if quantization_config is None:
            quantization_config = HFIntegration.make_quantization_config()

        if from_gguf or model_name.endswith(".gguf"):
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                gguf_file=model_name if model_name.endswith(".gguf") else None,
                **kwargs,
            )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            **kwargs,
        )

    @staticmethod
    def load_tokenizer(
        model_name: str, **kwargs: Any
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast | None:
        r"""Load a Hugging Face tokenizer.

        Args:
            model_name (str): repo id or local path.
            **kwargs: extra keyword arguments forwarded to
                :meth:`transformers.AutoTokenizer.from_pretrained`.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast | None: loaded tokenizer.
        """
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required for tokenizer loading. Install qwendopamine[hf]."
            )
        return AutoTokenizer.from_pretrained(model_name, **kwargs)

    @staticmethod
    def save_model(model: Any, save_directory: str) -> None:
        r"""Save a model with ``save_pretrained``.

        Args:
            model (Any): Hugging Face model instance.
            save_directory (str): output directory path.
        """
        model.save_pretrained(save_directory)
