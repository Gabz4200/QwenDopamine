"""Registration helpers for QwenDopamine HF architectures.

Register GDN2, Qwen3.5, and InfiniDopamine configs/models with HuggingFace
``AutoConfig`` / ``AutoModel`` / ``AutoModelForCausalLM``.
"""

from __future__ import annotations

from typing import Any

from qwendopamine.integrations.huggingface.configs import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    GDN2HFConfig,
)


def register_gdn2_hf() -> None:
    r"""Register GDN2 HF configs with HuggingFace ``AutoConfig``.

    Registers ``GDN2HFConfig``, ``Qwen35GDN2HFConfig``, and
    ``InfiniDopamineGDN2HFConfig`` under the names ``"gdn2"``,
    ``"qwen35_gdn2"``, and ``"infinidopamine_gdn2"``.

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.register_gdn2_hf` static method body.
    """
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


def register_qwen35_hf() -> None:
    r"""Register Qwen3.5 configs and models with HuggingFace Auto classes.

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.register_qwen35_hf` static method body.
    """
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

    if AutoModelForCausalLM is not None and hasattr(AutoModelForCausalLM, "register"):
        AutoModelForCausalLM.register(
            Qwen3_5TextConfig, Qwen3_5ForCausalLM, exist_ok=True
        )

    try:
        import transformers

        auto_cg: Any = getattr(transformers, "AutoModelForConditionalGeneration", None)
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


def register_infinidopamine_hf() -> None:
    r"""Register InfiniDopamine configs and models with HuggingFace Auto classes.

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.register_infinidopamine_hf` static method body.
    """
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

    if AutoModelForCausalLM is not None and hasattr(AutoModelForCausalLM, "register"):
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


def register_all_hf() -> None:
    r"""Register all QwenDopamine modules with HuggingFace Auto classes.

    .. note:: This is a direct copy of the original
        :meth:`HFIntegration.register_all_hf` static method body.
    """
    register_gdn2_hf()
    register_qwen35_hf()
    register_infinidopamine_hf()
