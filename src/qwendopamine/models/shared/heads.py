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
# limitations under the License.
"""Family-specific heads: causal LM, token classification, conditional generation, and sequence classification."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from transformers.utils import can_return_tuple

from qwendopamine.models.shared.outputs import FamilyModelOutputWithPast
from qwendopamine.models.shared.pretrained import FamilyPreTrainedModel


class FamilyForCausalLM(Qwen3ForCausalLM):
    r"""Base for family-specific ``ForCausalLM`` subclasses."""

    config_class: type
    config: Any
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
        r"^model.visual.*",
    ]

    def __init__(self, config: Any) -> None:
        r"""__init__(self, config: Any) -> None

        Build the causal LM head and underlying text model.

        Args:
            self - .
            config (Any) - .
        """
        if hasattr(config, "text_config") and not hasattr(config, "vocab_size"):
            config = config.text_config
        super().__init__(config)
        self.model = self._build_causal_lm_model(config)

    def _build_causal_lm_model(self, config: Any) -> nn.Module | None:
        r"""_build_causal_lm_model(self, config: Any) -> nn.Module | None

        Override to return family-specific text model.

        Args:
            self - .
            config (Any) - .

        Returns:
            nn.Module | None - .
        """

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""get_gate_regularization_loss(self, target: float=0.5) -> torch.Tensor

        Compute total gate balance regularization loss across all GDN-2 mixer layers.

        Args:
            self - .
            target (float) - .

        Returns:
            torch.Tensor - .
        """
        result: torch.Tensor = self.model.get_gate_regularization_loss(target=target)
        return result

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        reward_values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        r"""forward(self, input_ids: torch.LongTensor | None=None, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values: Cache | None=None, inputs_embeds: torch.FloatTensor | None=None, labels: torch.LongTensor | None=None, use_cache: bool | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, reward_values: torch.Tensor | None=None, **kwargs: Any) -> CausalLMOutputWithPast

        Compute causal LM logits and optional loss.

        Args:
            self - .
            input_ids (torch.LongTensor | None) - .
            attention_mask (torch.Tensor | None) - .
            position_ids (torch.LongTensor | None) - .
            past_key_values (Cache | None) - .
            inputs_embeds (torch.FloatTensor | None) - .
            labels (torch.LongTensor | None) - .
            use_cache (bool | None) - .
            output_attentions (bool | None) - .
            output_hidden_states (bool | None) - .
            return_dict (bool | None) - .
            reward_values (torch.Tensor | None) - .
            kwargs (Any) - .

        Returns:
            CausalLMOutputWithPast - .
        """
        outputs: CausalLMOutputWithPast = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            reward_values=reward_values,
            **kwargs,
        )
        self._apply_causal_lm_postprocessing(outputs)
        return outputs

    def _apply_causal_lm_postprocessing(self, outputs: CausalLMOutputWithPast) -> None:
        r"""_apply_causal_lm_postprocessing(self, outputs: CausalLMOutputWithPast) -> None

        Override to apply family-specific post-processing to causal LM outputs.

        Args:
            self - .
            outputs (CausalLMOutputWithPast) - .
        """


class FamilyForTokenClassification(
    GenericForTokenClassification, FamilyPreTrainedModel
):
    r"""Base for family-specific ``ForTokenClassification`` subclasses."""

    config_class: type
    config: Any


class FamilyForConditionalGeneration(Qwen3VLForConditionalGeneration):
    r"""Base for family-specific ``ForConditionalGeneration`` subclasses."""

    config_class: type
    config: Any
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
    ]

    def __init__(self, config: Any) -> None:
        r"""__init__(self, config: Any) -> None

        Initialize the conditional generation model and LM head.

        Args:
            self - .
            config (Any) - .
        """
        super().__init__(config)
        self.model = self._build_conditional_model(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    def _build_conditional_model(self, config: Any) -> nn.Module | None:
        r"""_build_conditional_model(self, config: Any) -> nn.Module | None

        Override to return family-specific model.

        Args:
            self - .
            config (Any) - .

        Returns:
            nn.Module | None - .
        """

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **super_kwargs: Any,
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        r"""get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor | None=None, **super_kwargs: Any) -> tuple[Any, ...] | BaseModelOutputWithPooling

        Delegate video feature extraction to the visual backbone.

        Args:
            self - .
            pixel_values_videos (torch.FloatTensor) - .
            video_grid_thw (torch.LongTensor | None) - .
            super_kwargs (Any) - .

        Returns:
            tuple[Any, ...] | BaseModelOutputWithPooling - .
        """
        return super().get_video_features(  # type: ignore[call-arg]
            pixel_values_videos,
            video_grid_thw=video_grid_thw,
            **super_kwargs,
        )

    def get_image_features(
        self, **super_kwargs: Any
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        r"""get_image_features(self, **super_kwargs: Any) -> tuple[Any, ...] | BaseModelOutputWithPooling

        Delegate image feature extraction to the visual backbone.

        Args:
            self - .
            super_kwargs (Any) - .

        Returns:
            tuple[Any, ...] | BaseModelOutputWithPooling - .
        """
        return super().get_image_features(**super_kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        reward_values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple | FamilyModelOutputWithPast:
        r"""forward(self, input_ids: torch.LongTensor | None=None, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values: Cache | None=None, inputs_embeds: torch.FloatTensor | None=None, labels: torch.LongTensor | None=None, pixel_values: torch.Tensor | None=None, pixel_values_videos: torch.FloatTensor | None=None, image_grid_thw: torch.LongTensor | None=None, video_grid_thw: torch.LongTensor | None=None, mm_token_type_ids: torch.IntTensor | None=None, logits_to_keep: int | torch.Tensor=0, reward_values: torch.Tensor | None=None, **kwargs: Any) -> tuple | FamilyModelOutputWithPast

        Compute conditional generation logits and optional loss.

        Args:
            self - .
            input_ids (torch.LongTensor | None) - .
            attention_mask (torch.Tensor | None) - .
            position_ids (torch.LongTensor | None) - .
            past_key_values (Cache | None) - .
            inputs_embeds (torch.FloatTensor | None) - .
            labels (torch.LongTensor | None) - .
            pixel_values (torch.Tensor | None) - .
            pixel_values_videos (torch.FloatTensor | None) - .
            image_grid_thw (torch.LongTensor | None) - .
            video_grid_thw (torch.LongTensor | None) - .
            mm_token_type_ids (torch.IntTensor | None) - .
            logits_to_keep (int | torch.Tensor) - .
            reward_values (torch.Tensor | None) - .
            kwargs (Any) - .

        Returns:
            tuple | FamilyModelOutputWithPast - .
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            reward_values=reward_values,
            **kwargs,
        )

        hidden_states = outputs[0]

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )

        self._apply_conditional_postprocessing(loss, outputs)
        if loss is not None:
            outputs.loss = loss

        return FamilyModelOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def _apply_conditional_postprocessing(
        self,
        loss: Any,
        outputs: Any,
    ) -> None:
        r"""_apply_conditional_postprocessing(self, loss: Any, outputs: Any) -> None

        Override to apply family-specific loss post-processing.

        Args:
            self - .
            loss (Any) - .
            outputs (Any) - .
        """

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""get_gate_regularization_loss(self, target: float=0.5) -> torch.Tensor

        Compute total gate balance regularization loss across all GDN-2 mixer layers.

        Args:
            self - .
            target (float) - .

        Returns:
            torch.Tensor - .
        """
        result: torch.Tensor = self.model.get_gate_regularization_loss(target=target)
        return result


class FamilyForSequenceClassification(
    GenericForSequenceClassification, FamilyPreTrainedModel
):
    r"""Base for family-specific ``ForSequenceClassification`` subclasses."""

    config_class: type
    config: Any

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutputWithPast:
        r"""forward(self, input_ids: torch.LongTensor | None=None, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values: Cache | None=None, inputs_embeds: torch.FloatTensor | None=None, pixel_values: torch.Tensor | None=None, pixel_values_videos: torch.FloatTensor | None=None, image_grid_thw: torch.LongTensor | None=None, video_grid_thw: torch.LongTensor | None=None, mm_token_type_ids: torch.IntTensor | None=None, **kwargs: Any) -> SequenceClassifierOutputWithPast

        Run sequence classification over multimodal inputs.

        Args:
            self - .
            input_ids (torch.LongTensor | None) - .
            attention_mask (torch.Tensor | None) - .
            position_ids (torch.LongTensor | None) - .
            past_key_values (Cache | None) - .
            inputs_embeds (torch.FloatTensor | None) - .
            pixel_values (torch.Tensor | None) - .
            pixel_values_videos (torch.FloatTensor | None) - .
            image_grid_thw (torch.LongTensor | None) - .
            video_grid_thw (torch.LongTensor | None) - .
            mm_token_type_ids (torch.IntTensor | None) - .
            kwargs (Any) - .

        Returns:
            SequenceClassifierOutputWithPast - .
        """
        result: SequenceClassifierOutputWithPast = super().forward(  # type: ignore[assignment]
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        return result
