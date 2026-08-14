"""Behavioral tests for Hugging Face integrations."""

from __future__ import annotations

import torch

from qwendopamine.integrations.huggingface import HFIntegration


def test_when_make_quantization_config_int8_then_returns_bitsandbytes_config() -> None:
    qconfig = HFIntegration.make_quantization_config(method="int8", device="cpu")
    assert getattr(qconfig, "load_in_8bit", False) is True


def test_when_make_quantization_config_int4_then_returns_4bit_config() -> None:
    qconfig = HFIntegration.make_quantization_config(method="int4", compute_dtype="bfloat16")
    assert getattr(qconfig, "load_in_4bit", False) is True
    assert getattr(qconfig, "bnb_4bit_compute_dtype", None) == torch.bfloat16
