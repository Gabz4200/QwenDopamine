# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.
"""PyTorch Qwen3.5 model."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from qwendopamine.models._hf_compat import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    Cache,
    causal_conv1d_fn,
    causal_conv1d_update,
    init,
    logging,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
    unwrap_gated_delta_rule_fns,
    use_kernel_forward_from_hub,
    use_kernelized_func,
)

unwrap_gated_delta_rule_fns()

logger = logging.get_logger(__name__)


from qwendopamine.models.qwen35.configs import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from qwendopamine.models.qwen35.decoder_layer import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
)
from qwendopamine.models.qwen35.rotary_embeddings import (
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionRotaryEmbedding,
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


@use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")
@use_kernelized_func(
    [
        torch_recurrent_gated_delta_rule,
        torch_chunk_gated_delta_rule,
        causal_conv1d_fn,
        causal_conv1d_update,
    ]
)
class Qwen3_5PreTrainedModel(FamilyPreTrainedModel):
    config_class = Qwen3_5Config
    config: Qwen3_5Config
    _no_split_modules: ClassVar[list[str]] = [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
    ]
    _can_record_outputs: ClassVar[dict[str, type]] = {
        "hidden_states": Qwen3_5DecoderLayer,
        "attentions": Qwen3_5Attention,
    }

    def _init_family_weights(self, module: nn.Module) -> None:
        if isinstance(module, Qwen3_5GatedDeltaNet):
            init.ones_(module.dt_bias)
            # Lower bound kept away from 0 so log(A) never becomes -inf
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device)
                .uniform_(0.01, 16)
                .log_(),
            )
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, Qwen3_5RMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen3_5VisionRotaryEmbedding):
            inv_freq = 1.0 / (
                module.theta
                ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim)
            )
            init.copy_(module.inv_freq, inv_freq)


class Qwen3_5VisionModel(FamilyVisionModel):
    config_class = Qwen3_5VisionConfig
    config: Qwen3_5VisionConfig
    _no_split_modules: ClassVar[list[str]] = ["Qwen3_5VisionBlock"]

    def _delete_vision_attributes(self) -> None:
        del self.deepstack_visual_indexes
        del self.deepstack_merger_list


class Qwen3_5ModelOutputWithPast(FamilyModelOutputWithPast):
    pass


class Qwen3_5TextModel(FamilyTextModel):
    config_class = Qwen3_5TextConfig
    config: Qwen3_5TextConfig

    def _build_text_layers(self, config: Qwen3_5TextConfig) -> None:
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

    def _build_text_output(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None,
    ) -> BaseModelOutputWithPast:
        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3_5Model(FamilyModel):
    config_class = Qwen3_5Config
    _no_split_modules: ClassVar[list[str]] = [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
    ]

    def _build_model_components(self, config: Qwen3_5Config) -> None:
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)


class Qwen3_5ForCausalLM(FamilyForCausalLM):
    config_class = Qwen3_5TextConfig
    config: Qwen3_5TextConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
        r"^model.visual.*",
    ]

    def _build_causal_lm_model(self, config: Qwen3_5TextConfig) -> nn.Module:
        return Qwen3_5TextModel(config)


class Qwen3_5ForTokenClassification(FamilyForTokenClassification):
    config_class = Qwen3_5Config
    config: Qwen3_5Config


class Qwen3_5ForConditionalGeneration(FamilyForConditionalGeneration):
    config_class = Qwen3_5Config
    config: Qwen3_5Config
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
    ]

    def _build_conditional_model(self, config: Qwen3_5Config) -> nn.Module:
        return Qwen3_5Model(config)

    def get_video_features(
        self, **super_kwargs
    ) -> tuple | BaseModelOutputWithPooling:
        return super().get_video_features(**super_kwargs)

    def get_image_features(
        self, **super_kwargs
    ) -> tuple | BaseModelOutputWithPooling:
        return super().get_image_features(**super_kwargs)


class Qwen3_5TextForSequenceClassification(FamilyTextForSequenceClassification):
    config_class = Qwen3_5TextConfig
    config: Qwen3_5TextConfig
    input_modalities = ("text",)


class Qwen3_5ForSequenceClassification(FamilyForSequenceClassification):
    config_class = Qwen3_5Config
    config: Qwen3_5Config


__all__ = [
    "Qwen3_5Config",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForSequenceClassification",
    "Qwen3_5ForTokenClassification",
    "Qwen3_5PreTrainedModel",
    "Qwen3_5TextConfig",
    "Qwen3_5TextForSequenceClassification",
    "Qwen3_5TextModel",
    "Qwen3_5VisionConfig",
    "Qwen3_5VisionModel",
]
