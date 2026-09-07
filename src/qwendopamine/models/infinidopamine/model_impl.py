# Copyright 2025 The Qwen Team, InfiniDopamine Authors, and The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the license.
"""InfiniDopamine model implementation classes."""

from __future__ import annotations

from typing import ClassVar

from torch import nn
from transformers import initialization as init

from qwendopamine.models.infinidopamine.configs import (
    InfiniDopamineConfig,
    InfiniDopamineTextConfig,
    InfiniDopamineVisionConfig,
)
from qwendopamine.models.shared.model_family import (
    FamilyForCausalLM,
    FamilyForConditionalGeneration,
    FamilyForSequenceClassification,
    FamilyForTokenClassification,
    FamilyModel,
    FamilyModelOutputWithPast,
    FamilyPreTrainedModel,
    FamilyTextForSequenceClassification,
    FamilyTextModel,
    FamilyVisionModel,
)


class InfiniDopaminePreTrainedModel(FamilyPreTrainedModel):
    r"""InfiniDopaminePreTrainedModel: base pretrained model with GDN-2 weight
    initialization.

    Args:
        config (InfiniDopamineConfig): Model configuration.
    """

    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
    _no_split_modules: ClassVar[list[str]] = [
        "InfiniDopamineDecoderLayer",
        "InfiniDopamineVisionBlock",
    ]

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.trunc_normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)


class InfiniDopamineVisionModel(FamilyVisionModel):
    r"""InfiniDopamineVisionModel: vision encoder for multimodal configs.

    Args:
        config (InfiniDopamineVisionConfig): Vision configuration.
    """

    config_class = InfiniDopamineVisionConfig
    config: InfiniDopamineVisionConfig
    _no_split_modules: ClassVar[list[str]] = ["InfiniDopamineVisionBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.trunc_normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)


class InfiniDopamineModelOutputWithPast(FamilyModelOutputWithPast):
    r"""InfiniDopamineModelOutputWithPast: output container with past-key states."""



class InfiniDopamineTextModel(FamilyTextModel):
    r"""InfiniDopamineTextModel: text-only decoder stack with GDN-2 layers.

    Args:
        config (InfiniDopamineTextConfig): Text configuration.
    """

    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig

    def _build_text_layers(self, config: InfiniDopamineTextConfig) -> None:
        """Build text layers."""


class InfiniDopamineModel(FamilyModel):
    r"""InfiniDopamineModel: multimodal model combining vision and text towers.

    Args:
        config (InfiniDopamineConfig): Multimodal configuration.
    """

    config_class = InfiniDopamineConfig
    _no_split_modules: ClassVar[list[str]] = [
        "InfiniDopamineDecoderLayer",
        "InfiniDopamineVisionBlock",
    ]

    def __init__(self, config: InfiniDopamineConfig) -> None:
        super().__init__(config)

        """Build vision and text submodels."""
        self.visual = InfiniDopamineVisionModel(config.vision_config)
        self.language_model = InfiniDopamineTextModel(config.text_config)


class InfiniDopamineForCausalLM(FamilyForCausalLM):
    r"""InfiniDopamineForCausalLM: causal LM head with optional gate/parallel-loss.

    Args:
        config (InfiniDopamineTextConfig): Text configuration.
    """

    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
    ]

    def __init__(self, config: InfiniDopamineTextConfig) -> None:
        super().__init__(config)

        self.language_model = InfiniDopamineTextModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)


class InfiniDopamineForTokenClassification(FamilyForTokenClassification):
    r"""InfiniDopamineForTokenClassification: token-classification head.

    Args:
        config (InfiniDopamineConfig): Model configuration.
    """

    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig


class InfiniDopamineForConditionalGeneration(FamilyForConditionalGeneration):
    r"""InfiniDopamineForConditionalGeneration: multimodal causal-LM head.

    Args:
        config (InfiniDopamineConfig): Multimodal configuration.
    """

    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
    ]

    def __init__(self, config: InfiniDopamineConfig) -> None:
        super().__init__(config)

        self.text_model = InfiniDopamineTextModel(config.text_config)


class InfiniDopamineTextForSequenceClassification(FamilyTextForSequenceClassification):
    r"""InfiniDopamineTextForSequenceClassification: sequence classification on
    text input.

    Args:
        config (InfiniDopamineTextConfig): Text configuration.
    """

    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig
    input_modalities = ("text",)


class InfiniDopamineForSequenceClassification(FamilyForSequenceClassification):
    r"""InfiniDopamineForSequenceClassification: sequence classification on
    multimodal input.

    Args:
        config (InfiniDopamineConfig): Multimodal configuration.
    """

    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig