"""High-level HF integration facade for QwenDopamine models.

:class:`HFIntegration` is a thin static-method facade over the per-concern
helper modules:

  - :mod:`.registration` — ``register_*_hf`` family
  - :mod:`.loading`      — ``load_config`` / ``load_model`` / ``load_tokenizer``
  - :mod:`.quantization` — ``make_quantization_config``
  - :mod:`._build`       — ``build_*`` and ``prepare_model_for_trl_training``
  - :mod:`._save`        — ``save_model``

All methods are static; import the class and call the methods directly:

.. code-block:: python

    from qwendopamine.integrations.huggingface import HFIntegration
    HFIntegration.register_infinidopamine_hf()
"""

from __future__ import annotations

from typing import Any

from qwendopamine.integrations.huggingface._build import (
    build_gdn2_hf_block,
    build_gdn2_hf_config,
    build_infinidopamine_config,
    build_infinidopamine_model,
    prepare_model_for_trl_training,
)
from qwendopamine.integrations.huggingface._save import save_model
from qwendopamine.integrations.huggingface.loading import (
    load_config,
    load_model,
    load_tokenizer,
)
from qwendopamine.integrations.huggingface.quantization import (
    make_quantization_config,
)
from qwendopamine.integrations.huggingface.registration import (
    register_all_hf,
    register_gdn2_hf,
    register_infinidopamine_hf,
    register_qwen35_hf,
)


class HFIntegration:
    r"""Facade for HuggingFace ``AutoConfig``, ``AutoModel``, and tokenizer
    registration of QwenDopamine architectures.

    All methods are static; import the class and call the methods directly.
    """

    @staticmethod
    def register_gdn2_hf() -> None:
        return register_gdn2_hf()

    @staticmethod
    def register_qwen35_hf() -> None:
        return register_qwen35_hf()

    @staticmethod
    def register_infinidopamine_hf() -> None:
        return register_infinidopamine_hf()

    @staticmethod
    def register_all_hf() -> None:
        return register_all_hf()

    @staticmethod
    def build_infinidopamine_config(
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        sliding_window: int = 1024,
        **kwargs: Any,
    ) -> Any:
        return build_infinidopamine_config(
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
        return build_infinidopamine_model(config=config, **kwargs)

    @staticmethod
    def prepare_model_for_trl_training(
        model: Any,
        use_gradient_checkpointing: bool = True,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return prepare_model_for_trl_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )

    @staticmethod
    def build_gdn2_hf_config(
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        **kwargs: Any,
    ) -> Any:
        return build_gdn2_hf_config(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            **kwargs,
        )

    @staticmethod
    def build_gdn2_hf_block(
        config: Any,
        layer_idx: int,
    ) -> Any:
        return build_gdn2_hf_block(config=config, layer_idx=layer_idx)

    @staticmethod
    def make_quantization_config(
        method: str = "int8",
        compute_dtype: str = "bfloat16",
        device: str = "cpu",
    ) -> Any:
        return make_quantization_config(
            method=method, compute_dtype=compute_dtype, device=device
        )

    @staticmethod
    def load_config(model_name: str, **kwargs: Any) -> Any:
        return load_config(model_name, **kwargs)

    @staticmethod
    def load_model(
        model_name: str,
        **kwargs: Any,
    ) -> Any:
        return load_model(model_name, **kwargs)

    @staticmethod
    def load_tokenizer(
        model_name: str,
        **kwargs: Any,
    ) -> Any:
        return load_tokenizer(model_name, **kwargs)

    @staticmethod
    def save_model(model: Any, save_directory: str) -> None:
        return save_model(model, save_directory)


__all__ = ["HFIntegration"]
