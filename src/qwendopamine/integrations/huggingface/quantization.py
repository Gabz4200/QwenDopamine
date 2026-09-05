"""Quantization config helpers for QwenDopamine HF models.

Provides ``make_quantization_config`` that builds ``BitsAndBytesConfig`` or
``QuantoConfig`` based on the requested method string.
"""

from __future__ import annotations

import torch

from qwendopamine.integrations.huggingface.configs import (
    BitsAndBytesConfig,
    QuantoConfig,
)


def make_quantization_config(
    method: str = "int8", compute_dtype: str = "bfloat16", device: str = "cpu"
) -> BitsAndBytesConfig | QuantoConfig:
    r"""Build a Hugging Face quantization config for CPU-friendly loading.

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.make_quantization_config` static method body.

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
