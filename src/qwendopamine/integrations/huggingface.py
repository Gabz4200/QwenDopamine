from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, QuantoConfig


class HFIntegration:
    @staticmethod
    def make_quantization_config(method: str = "int8", compute_dtype: str = "bfloat16", device: str = "cpu") -> Any:
        if method == "int8":
            dtype = getattr(torch, compute_dtype, torch.bfloat16)
            return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=device == "cpu")
        if method == "int4":
            dtype = getattr(torch, compute_dtype, torch.bfloat16)
            return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        return QuantoConfig(weights=method)

    @staticmethod
    def load_config(model_name: str, **kwargs: Any):
        return AutoConfig.from_pretrained(model_name, **kwargs)

    @staticmethod
    def load_model(model_name: str, quantization_config: Any = None, device_map: str = "cpu", from_gguf: bool = False, **kwargs: Any):
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
    def load_tokenizer(model_name: str, **kwargs: Any):
        return AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def save_model(model: Any, save_directory: str) -> None:
        model.save_pretrained(save_directory)
