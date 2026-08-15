from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import PreTrainedConfig as _BaseConfig

    from qwendopamine.models.gdn2.config import GDN2Config
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
    def from_gdn2_config(
        cls, config: GDN2Config | dict[str, Any] | Any, **kwargs: Any
    ) -> GDN2HFConfig:
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

    def to_gdn2_config(self) -> GDN2Config:
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


class GatedSurpriseNetHFConfig(_BaseConfig):
    r"""Hugging Face PreTrainedConfig adapter for GatedSurpriseNetAdam configuration."""

    model_type = "gated_surprise_net"

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
        local_adam_lr: float = 1e-3,
        local_adam_beta1: float = 0.9,
        local_adam_beta2: float = 0.999,
        local_adam_eps: float = 1e-8,
        nll_var_eps: float = 1e-6,
        nll_full: bool = False,
        learnable_init_state: bool = False,
        train_chunk_size: int = 128,
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
        self.local_adam_lr = local_adam_lr
        self.local_adam_beta1 = local_adam_beta1
        self.local_adam_beta2 = local_adam_beta2
        self.local_adam_eps = local_adam_eps
        self.nll_var_eps = nll_var_eps
        self.nll_full = nll_full
        self.learnable_init_state = learnable_init_state
        self.train_chunk_size = train_chunk_size
        self.vocab_size = vocab_size
        super().__init__(**kwargs)


class Qwen35GatedSurpriseNetHFConfig(GatedSurpriseNetHFConfig):
    r"""Hugging Face PreTrainedConfig adapter for Qwen3.5 GatedSurpriseNet variant."""

    model_type = "qwen35_gated_surprise_net"


class GDN2HFBlock(nn.Module):
    r"""Hugging Face compatible nn.Module block wrapper around GatedDeltaNet2."""

    def __init__(
        self,
        config: PreTrainedConfig | GDN2Config | dict[str, Any] | Any,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from qwendopamine.models.gdn2.gdn2 import GatedDeltaNet2

        if isinstance(config, GDN2HFConfig):
            self.config: GDN2HFConfig = config
        else:
            self.config = GDN2HFConfig.from_gdn2_config(config, **kwargs)
        self.layer_idx = layer_idx
        self.mixer = GatedDeltaNet2(self.config, layer_idx=layer_idx, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        return self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )


class GatedSurpriseNetHFBlock(nn.Module):
    r"""Hugging Face compatible nn.Module block wrapper around GatedSurpriseNetAdam."""

    def __init__(
        self,
        config: PreTrainedConfig | dict[str, Any] | Any,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from qwendopamine.models.gated_surprise_net import GatedSurpriseNetAdam

        if isinstance(config, GatedSurpriseNetHFConfig):
            self.config: GatedSurpriseNetHFConfig = config
        elif isinstance(config, dict):
            self.config = GatedSurpriseNetHFConfig(**config)
        else:
            attrs = {
                k: getattr(config, k)
                for k in [
                    "hidden_size",
                    "num_heads",
                    "head_dim",
                    "num_v_heads",
                    "expand_v",
                    "conv_size",
                    "conv_bias",
                    "allow_neg_eigval",
                    "norm_eps",
                    "local_adam_lr",
                    "local_adam_beta1",
                    "local_adam_beta2",
                    "local_adam_eps",
                    "nll_var_eps",
                    "nll_full",
                    "learnable_init_state",
                    "train_chunk_size",
                ]
                if hasattr(config, k)
            }
            self.config = GatedSurpriseNetHFConfig(**attrs)

        self.layer_idx = layer_idx
        kwargs_dict = {
            "hidden_size": getattr(self.config, "hidden_size", 2048),
            "num_heads": getattr(self.config, "num_heads", 16),
            "head_dim": getattr(self.config, "head_dim", 128),
            "num_v_heads": getattr(self.config, "num_v_heads", None),
            "expand_v": getattr(self.config, "expand_v", 1.0),
            "conv_size": getattr(self.config, "conv_size", 4),
            "conv_bias": getattr(self.config, "conv_bias", False),
            "allow_neg_eigval": getattr(self.config, "allow_neg_eigval", False),
            "norm_eps": getattr(self.config, "norm_eps", 1e-5),
            "local_adam_lr": getattr(self.config, "local_adam_lr", 1e-3),
            "local_adam_beta1": getattr(self.config, "local_adam_beta1", 0.9),
            "local_adam_beta2": getattr(self.config, "local_adam_beta2", 0.999),
            "local_adam_eps": getattr(self.config, "local_adam_eps", 1e-8),
            "nll_var_eps": getattr(self.config, "nll_var_eps", 1e-6),
            "nll_full": getattr(self.config, "nll_full", False),
            "learnable_init_state": getattr(self.config, "learnable_init_state", False),
            "train_chunk_size": getattr(self.config, "train_chunk_size", 128),
        }
        kwargs_dict.update(kwargs)
        self.mixer = GatedSurpriseNetAdam(layer_idx=layer_idx, **kwargs_dict)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        return self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )


class HFIntegration:
    @staticmethod
    def register_gdn2_hf() -> None:
        r"""Register GDN2HFConfig, Qwen35GDN2HFConfig, GatedSurpriseNetHFConfig with AutoConfig."""
        if AutoConfig is not None and hasattr(AutoConfig, "register"):
            AutoConfig.register("gdn2", GDN2HFConfig, exist_ok=True)
            AutoConfig.register("qwen35_gdn2", Qwen35GDN2HFConfig, exist_ok=True)
            AutoConfig.register(
                "gated_surprise_net", GatedSurpriseNetHFConfig, exist_ok=True
            )
            AutoConfig.register(
                "qwen35_gated_surprise_net",
                Qwen35GatedSurpriseNetHFConfig,
                exist_ok=True,
            )

    @staticmethod
    def build_gdn2_hf_config(
        config_or_name: str | GDN2Config | dict[str, Any] | Any = "gdn2_1.3B",
        **kwargs: Any,
    ) -> GDN2HFConfig:
        r"""Build a GDN2HFConfig instance from a name or config object."""
        from qwendopamine.models.gdn2.config import GDN2Config

        if isinstance(config_or_name, str):
            cfg = GDN2Config.from_name(config_or_name, **kwargs)
            return GDN2HFConfig.from_gdn2_config(cfg)
        return GDN2HFConfig.from_gdn2_config(config_or_name, **kwargs)

    @staticmethod
    def build_gdn2_hf_block(
        config: PreTrainedConfig | GDN2Config | dict[str, Any] | Any,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> GDN2HFBlock:
        r"""Build a Hugging Face compatible GDN2HFBlock module."""
        return GDN2HFBlock(config=config, layer_idx=layer_idx, **kwargs)

    @staticmethod
    def build_gated_surprise_net_hf_config(
        config_or_dict: dict[str, Any] | Any = None,
        **kwargs: Any,
    ) -> GatedSurpriseNetHFConfig:
        r"""Build a GatedSurpriseNetHFConfig instance."""
        data: dict[str, Any] = {}
        if isinstance(config_or_dict, dict):
            data.update(config_or_dict)
        elif config_or_dict is not None:
            data.update(
                {
                    k: getattr(config_or_dict, k)
                    for k in [
                        "hidden_size",
                        "num_heads",
                        "head_dim",
                        "num_v_heads",
                        "expand_v",
                        "conv_size",
                        "conv_bias",
                        "allow_neg_eigval",
                        "norm_eps",
                        "local_adam_lr",
                        "local_adam_beta1",
                        "local_adam_beta2",
                        "local_adam_eps",
                        "nll_var_eps",
                        "nll_full",
                        "learnable_init_state",
                        "train_chunk_size",
                    ]
                    if hasattr(config_or_dict, k)
                }
            )
        data.update(kwargs)
        return GatedSurpriseNetHFConfig(**data)

    @staticmethod
    def build_gated_surprise_net_hf_block(
        config: PreTrainedConfig | dict[str, Any] | Any,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> GatedSurpriseNetHFBlock:
        r"""Build a Hugging Face compatible GatedSurpriseNetHFBlock module."""
        return GatedSurpriseNetHFBlock(config=config, layer_idx=layer_idx, **kwargs)

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
