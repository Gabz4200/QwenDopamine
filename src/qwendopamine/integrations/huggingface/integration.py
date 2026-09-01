"""High-level HF integration facade for QwenDopamine models.

:class:`HFIntegration` groups registration helpers (AutoConfig / AutoModel /
AutoModelForCausalLM) and load / save / quantize entry points. All methods are
static so callers can pick the ones they need without instantiating.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.integrations.huggingface.block import GDN2HFBlock
from qwendopamine.integrations.huggingface.configs import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GDN2HFConfig,
    PreTrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    QuantoConfig,
)


class HFIntegration:
    @staticmethod
    def register_gdn2_hf() -> None:
        r"""Register GDN2HFConfig, Qwen35GDN2HFConfig, and InfiniDopamineGDN2HFConfig with AutoConfig."""
        from qwendopamine.integrations.huggingface.configs import (
            InfiniDopamineGDN2HFConfig,
            Qwen35GDN2HFConfig,
        )

        if AutoConfig is not None and hasattr(AutoConfig, "register"):
            AutoConfig.register("gdn2", GDN2HFConfig, exist_ok=True)
            AutoConfig.register("qwen35_gdn2", Qwen35GDN2HFConfig, exist_ok=True)
            AutoConfig.register(
                "infinidopamine_gdn2", InfiniDopamineGDN2HFConfig, exist_ok=True
            )

    @staticmethod
    def register_qwen35_hf() -> None:
        r"""Register Qwen3.5 configs and models with Hugging Face Auto classes."""
        from qwendopamine.models.qwen35 import (
            Qwen3_5Config,
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5ForSequenceClassification,
            Qwen3_5ForTokenClassification,
            Qwen3_5Model,
            Qwen3_5TextConfig,
            Qwen3_5TextForSequenceClassification,
            Qwen3_5TextModel,
            Qwen3_5VisionConfig,
            Qwen3_5VisionModel,
        )

        if AutoConfig is not None and hasattr(AutoConfig, "register"):
            AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
            AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
            AutoConfig.register("qwen3_5_vision", Qwen3_5VisionConfig, exist_ok=True)

        if AutoModel is not None and hasattr(AutoModel, "register"):
            AutoModel.register(Qwen3_5TextConfig, Qwen3_5TextModel, exist_ok=True)
            AutoModel.register(Qwen3_5Config, Qwen3_5Model, exist_ok=True)
            AutoModel.register(Qwen3_5VisionConfig, Qwen3_5VisionModel, exist_ok=True)

        if AutoModelForCausalLM is not None and hasattr(
            AutoModelForCausalLM, "register"
        ):
            AutoModelForCausalLM.register(
                Qwen3_5TextConfig, Qwen3_5ForCausalLM, exist_ok=True
            )

        try:
            import transformers

            auto_cg: Any = getattr(
                transformers, "AutoModelForConditionalGeneration", None
            )
            if auto_cg is not None and hasattr(auto_cg, "register"):
                auto_cg.register(
                    Qwen3_5Config, Qwen3_5ForConditionalGeneration, exist_ok=True
                )
        except (ImportError, AttributeError):
            pass

        try:
            from transformers import AutoModelForSequenceClassification

            if AutoModelForSequenceClassification is not None and hasattr(
                AutoModelForSequenceClassification, "register"
            ):
                AutoModelForSequenceClassification.register(
                    Qwen3_5TextConfig,
                    Qwen3_5TextForSequenceClassification,
                    exist_ok=True,
                )
                AutoModelForSequenceClassification.register(
                    Qwen3_5Config, Qwen3_5ForSequenceClassification, exist_ok=True
                )
        except (ImportError, AttributeError):
            pass

        try:
            from transformers import AutoModelForTokenClassification

            if AutoModelForTokenClassification is not None and hasattr(
                AutoModelForTokenClassification, "register"
            ):
                AutoModelForTokenClassification.register(
                    Qwen3_5Config, Qwen3_5ForTokenClassification, exist_ok=True
                )
        except (ImportError, AttributeError):
            pass

    @staticmethod
    def register_infinidopamine_hf() -> None:
        r"""Register InfiniDopamine configs and models with Hugging Face Auto classes for TRL and Transformers compatibility."""
        from qwendopamine.models.infinidopamine import (
            InfiniDopamineConfig,
            InfiniDopamineForCausalLM,
            InfiniDopamineForConditionalGeneration,
            InfiniDopamineForSequenceClassification,
            InfiniDopamineForTokenClassification,
            InfiniDopamineModel,
            InfiniDopamineTextConfig,
            InfiniDopamineTextForSequenceClassification,
            InfiniDopamineTextModel,
            InfiniDopamineVisionConfig,
            InfiniDopamineVisionModel,
        )

        if AutoConfig is not None and hasattr(AutoConfig, "register"):
            AutoConfig.register("infinidopamine", InfiniDopamineConfig, exist_ok=True)
            AutoConfig.register(
                "infinidopamine_text", InfiniDopamineTextConfig, exist_ok=True
            )
            AutoConfig.register(
                "infinidopamine_vision", InfiniDopamineVisionConfig, exist_ok=True
            )

        if AutoModel is not None and hasattr(AutoModel, "register"):
            AutoModel.register(
                InfiniDopamineTextConfig, InfiniDopamineTextModel, exist_ok=True
            )
            AutoModel.register(InfiniDopamineConfig, InfiniDopamineModel, exist_ok=True)
            AutoModel.register(
                InfiniDopamineVisionConfig, InfiniDopamineVisionModel, exist_ok=True
            )

        if AutoModelForCausalLM is not None and hasattr(
            AutoModelForCausalLM, "register"
        ):
            AutoModelForCausalLM.register(
                InfiniDopamineTextConfig, InfiniDopamineForCausalLM, exist_ok=True
            )

        try:
            import transformers

            auto_cg_inf: Any = getattr(
                transformers, "AutoModelForConditionalGeneration", None
            )
            if auto_cg_inf is not None and hasattr(auto_cg_inf, "register"):
                auto_cg_inf.register(
                    InfiniDopamineConfig,
                    InfiniDopamineForConditionalGeneration,
                    exist_ok=True,
                )
        except (ImportError, AttributeError):
            pass

        try:
            from transformers import AutoModelForSequenceClassification

            if AutoModelForSequenceClassification is not None and hasattr(
                AutoModelForSequenceClassification, "register"
            ):
                AutoModelForSequenceClassification.register(
                    InfiniDopamineTextConfig,
                    InfiniDopamineTextForSequenceClassification,
                    exist_ok=True,
                )
                AutoModelForSequenceClassification.register(
                    InfiniDopamineConfig,
                    InfiniDopamineForSequenceClassification,
                    exist_ok=True,
                )
        except (ImportError, AttributeError):
            pass

        try:
            from transformers import AutoModelForTokenClassification

            if AutoModelForTokenClassification is not None and hasattr(
                AutoModelForTokenClassification, "register"
            ):
                AutoModelForTokenClassification.register(
                    InfiniDopamineConfig,
                    InfiniDopamineForTokenClassification,
                    exist_ok=True,
                )
        except (ImportError, AttributeError):
            pass

    @staticmethod
    def register_all_hf() -> None:
        r"""Register all QwenDopamine modules (GDN2, Qwen3.5, InfiniDopamine) with Hugging Face Auto classes."""
        HFIntegration.register_gdn2_hf()
        HFIntegration.register_qwen35_hf()
        HFIntegration.register_infinidopamine_hf()

    @staticmethod
    def build_infinidopamine_config(
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        sliding_window: int = 1024,
        **kwargs: Any,
    ) -> Any:
        r"""Build an InfiniDopamineTextConfig instance with sensible defaults."""
        from qwendopamine.models.infinidopamine import InfiniDopamineTextConfig

        return InfiniDopamineTextConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            sliding_window=sliding_window,
            **kwargs,
        )

    @staticmethod
    def build_infinidopamine_model(
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        r"""Build an InfiniDopamineForCausalLM model instance."""
        from qwendopamine.models.infinidopamine import (
            InfiniDopamineForCausalLM,
            InfiniDopamineTextConfig,
        )

        if config is None:
            text_cfg = InfiniDopamineTextConfig(**kwargs)
        elif isinstance(config, dict):
            text_cfg = InfiniDopamineTextConfig(**config)
        else:
            text_cfg = config
        return InfiniDopamineForCausalLM(text_cfg)

    @staticmethod
    def prepare_model_for_trl_training(
        model: Any,
        use_gradient_checkpointing: bool = True,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        r"""Prepare an InfiniDopamine or Qwen model for TRL training (SFTTrainer, DPOTrainer, GRPOTrainer).

        Enables gradient checkpointing, ensures input embeddings calculate gradients,
        and verifies generation / causal-LM training contracts.
        """
        if use_gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
                try:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=kwargs
                    )
                except TypeError:
                    model.gradient_checkpointing_enable()

            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def _make_inputs_require_grad(
                    module: nn.Module, input_t: torch.Tensor, output: torch.Tensor
                ) -> None:
                    output.requires_grad_(True)

                if hasattr(model, "get_input_embeddings"):
                    emb = model.get_input_embeddings()
                    if emb is not None:
                        emb.register_forward_hook(_make_inputs_require_grad)

        if hasattr(model, "config") and model.config is not None:
            model.config.use_cache = False

        return model

    @staticmethod
    def build_gdn2_hf_config(
        config_or_name: str | Any = "gdn2_1.3B",
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
        config: PreTrainedConfig | Any,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> GDN2HFBlock:
        r"""Build a Hugging Face compatible GDN2HFBlock module."""
        return GDN2HFBlock(config=config, layer_idx=layer_idx, **kwargs)

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


__all__ = ["HFIntegration"]