"""HF PreTrainedConfig adapters for QwenDopamine architectures.

Defines :class:`GDN2HFConfig`, :class:`Qwen35GDN2HFConfig`, and
:class:`InfiniDopamineGDN2HFConfig`, plus the optional-import fallbacks for
``transformers`` symbols used across the integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedConfig as _BaseConfig
else:
    try:
        from transformers import PreTrainedConfig as _BaseConfig
    except ModuleNotFoundError:

        class _BaseConfig:
            model_type: str = ""

            def __init__(self, **kwargs: Any) -> None:
                for k, v in kwargs.items():
                    setattr(self, k, v)


try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
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
    AutoModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = _FallbackBitsAndBytesConfig
    PreTrainedModel = Any
    PreTrainedTokenizer = Any
    PreTrainedTokenizerFast = Any
    QuantoConfig = _FallbackQuantoConfig


PreTrainedConfig = _BaseConfig


class GDN2HFConfig(_BaseConfig):
    r"""Hugging Face PreTrainedConfig adapter for GDN-2 module configuration."""

    model_type = "gdn2"

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        head_dim: int = 128,
        num_v_heads: int | None = None,
        expand_v: float = 1.0,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        block_size: int = 4096,
        vocab_size: int = 32000,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_v_heads = num_v_heads
        self.expand_v = expand_v
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps
        self.block_size = block_size
        self.vocab_size = vocab_size
        super().__init__(**kwargs)

    @classmethod
    def from_gdn2_config(cls, config: Any, **kwargs: Any) -> GDN2HFConfig:
        r"""Build a GDN2HFConfig instance from a GDN2Config or dict."""
        data: dict[str, Any] = {}
        if hasattr(config, "hidden_size"):
            data = {
                "hidden_size": int(getattr(config, "hidden_size", 2048)),
                "num_heads": int(getattr(config, "num_heads", 16)),
                "head_dim": int(getattr(config, "head_dim", 128)),
                "num_v_heads": getattr(config, "num_v_heads", None),
                "expand_v": float(getattr(config, "expand_v", 1.0)),
                "conv_size": int(getattr(config, "conv_size", 4)),
                "conv_bias": bool(getattr(config, "conv_bias", False)),
                "allow_neg_eigval": bool(getattr(config, "allow_neg_eigval", False)),
                "norm_eps": float(getattr(config, "norm_eps", 1e-5)),
                "block_size": int(getattr(config, "block_size", 4096)),
                "vocab_size": int(getattr(config, "vocab_size", 32000)),
            }
        elif isinstance(config, dict):
            data = dict(config)
        data.update(kwargs)
        return cls(**data)

    def to_gdn2_config(self) -> Any:
        r"""Convert back to GDN2Config dataclass."""
        from qwendopamine.models.gdn2.config import GDN2Config

        return GDN2Config(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_v_heads=self.num_v_heads,
            expand_v=self.expand_v,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            allow_neg_eigval=self.allow_neg_eigval,
            norm_eps=self.norm_eps,
            block_size=self.block_size,
            vocab_size=self.vocab_size,
        )


class Qwen35GDN2HFConfig(GDN2HFConfig):
    r"""Hugging Face PreTrainedConfig adapter for Qwen3.5 GDN-2 variant."""

    model_type = "qwen35_gdn2"


class InfiniDopamineGDN2HFConfig(GDN2HFConfig):
    r"""Hugging Face PreTrainedConfig adapter for InfiniDopamine GDN-2 variant."""

    model_type = "infinidopamine_gdn2"


__all__ = [
    "AutoConfig",
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoTokenizer",
    "BitsAndBytesConfig",
    "GDN2HFConfig",
    "InfiniDopamineGDN2HFConfig",
    "PreTrainedConfig",
    "PreTrainedModel",
    "PreTrainedTokenizer",
    "PreTrainedTokenizerFast",
    "QuantoConfig",
    "Qwen35GDN2HFConfig",
]
