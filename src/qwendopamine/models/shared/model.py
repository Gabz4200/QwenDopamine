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
"""Family-specific ``Model`` base class."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.utils import can_return_tuple

from qwendopamine.models.shared.outputs import FamilyModelOutputWithPast


class FamilyModel(Qwen3VLModel):
    r"""Base for family-specific ``Model`` subclasses.

    Subclasses must set ``config_class`` and optionally override
    ``_build_model_components`` to customize initialization.
    """

    config_class: type
    _no_split_modules: ClassVar[list[str]] = []

    def __init__(self, config: Any) -> None:
        r"""__init__(self, config: Any) -> None

        Initialize the multimodal model from config.

        Args:
            self - .
            config (Any) - .
        """
        super().__init__(config)
        self._build_model_components(config)
        self.rope_deltas = None
        self.post_init()

    def _build_model_components(self, config: Any) -> None:
        r"""_build_model_components(self, config: Any) -> None

        Override to add family-specific model components.

        Args:
            self - .
            config (Any) - .
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

    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        r"""get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None=None, **kwargs: Any) -> tuple[Any, ...] | BaseModelOutputWithPooling

        Encode images into visual embeddings.

        Args:
            self - .
            pixel_values (torch.FloatTensor) - .
            image_grid_thw (torch.LongTensor | None) - .
            kwargs (Any) - .

        Returns:
            tuple[Any, ...] | BaseModelOutputWithPooling - .
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output: BaseModelOutputWithPooling = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

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
    ) -> tuple[Any, ...] | FamilyModelOutputWithPast:
        r"""forward(self, input_ids: torch.LongTensor | None=None, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values: Cache | None=None, inputs_embeds: torch.FloatTensor | None=None, pixel_values: torch.Tensor | None=None, pixel_values_videos: torch.FloatTensor | None=None, image_grid_thw: torch.LongTensor | None=None, video_grid_thw: torch.LongTensor | None=None, mm_token_type_ids: torch.IntTensor | None=None, **kwargs: Any) -> tuple[Any, ...] | FamilyModelOutputWithPast

        Run the full multimodal model and merge vision/text streams.

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
            tuple[Any, ...] | FamilyModelOutputWithPast - .
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw=image_grid_thw, return_dict=True, **kwargs
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos,
                video_grid_thw=video_grid_thw,
                return_dict=True,
                **kwargs,
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return FamilyModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )
