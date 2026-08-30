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
"""Shared model family base classes for Qwen-style LLM architectures.

Both ``infinidopamine`` and ``qwen35`` families share the same HuggingFace
class hierarchy (``Qwen3NextPreTrainedModel`` -> ``Qwen3VLVisionModel`` ->
``Qwen3NextModel`` -> ``Qwen3VLModel`` -> ``Qwen3ForCausalLM`` /
``Qwen3VLForConditionalGeneration`` -> ``GenericForSequenceClassification`` /
``GenericForTokenClassification``).

This module extracts the duplicated boilerplate into reusable base classes so
the family-specific modules only need to declare their unique config classes,
layer types, and initialization hooks.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn

from qwendopamine.models._hf_compat import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    Cache,
    CausalLMOutputWithPast,
    DynamicCache,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    PreTrainedModel,
    Qwen3ForCausalLM,
    Qwen3NextModel,
    Qwen3NextPreTrainedModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLVisionModel,
    SequenceClassifierOutputWithPast,
    TransformersKwargs,
    Unpack,
    can_return_tuple,
    capture_outputs,
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
    expand_position_ids_to_multimodal,
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    merge_with_config_defaults,
)


class FamilyPreTrainedModel(Qwen3NextPreTrainedModel):
    """Base for family-specific ``PreTrainedModel`` subclasses."""

    _no_split_modules: ClassVar[list[str]] = []
    _can_record_outputs: ClassVar[dict[str, type]] = {}

    def _init_weights(self, module: nn.Module) -> None:
        PreTrainedModel._init_weights(self, module)
        self._init_family_weights(module)

    def _init_family_weights(self, module: nn.Module) -> None:
        """Override in subclasses to add family-specific weight initialization."""


class FamilyVisionModel(Qwen3VLVisionModel):
    """Base for family-specific ``VisionModel`` subclasses.

    Subclasses only need to set ``config_class``; the ``__init__`` and
    ``forward`` implementations are shared.
    """

    config_class: type
    config: Any
    _no_split_modules: ClassVar[list[str]] = []

    def __init__(self, config: Any, *inputs: Any, **kwargs: Any) -> None:
        super().__init__(config, *inputs, **kwargs)
        self._delete_vision_attributes()

    def _delete_vision_attributes(self) -> None:
        """Override to delete family-specific vision attributes after init."""

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.config.spatial_merge_size,
            kwargs=kwargs,
        )
        position_ids = get_vision_position_ids(
            grid_thw, self.spatial_merge_size, kwargs=kwargs
        )
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(
            grid_thw, self.config, kwargs=kwargs
        )
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (
            self.pos_embed(interp_indices) * interp_weights[:, :, None]
        ).sum(1)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class FamilyModelOutputWithPast(Qwen3VLModelOutputWithPast):
    """Empty pass-through for family-specific model output classes."""



class FamilyTextModel(Qwen3NextModel):
    """Base for family-specific ``TextModel`` subclasses.

    Subclasses must set ``config_class`` and optionally override
    ``_build_text_layers`` to customize layer initialization.
    """

    config_class: type
    config: Any

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._build_text_layers(config)

    def _build_text_layers(self, config: Any) -> None:
        """Override to add family-specific text layers after parent init."""

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            if past_key_values is not None:
                try:
                    past_seen_tokens = past_key_values.get_seq_length()
                except ValueError:
                    past_seen_tokens = 0
            else:
                past_seen_tokens = 0
        else:
            past_seen_tokens = 0

        position_ids, text_position_ids = expand_position_ids_to_multimodal(
            position_ids=position_ids,
            batch_size=inputs_embeds.shape[0],
            seq_len=inputs_embeds.shape[1],
            past_seen_tokens=past_seen_tokens,
            device=inputs_embeds.device,
        )

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            if getattr(self.config, "sliding_window", None) is not None:
                attn_mask = create_sliding_window_causal_mask(**mask_kwargs)
            else:
                attn_mask = create_causal_mask(**mask_kwargs)
            causal_mask_mapping = {
                "full_attention": attn_mask,
                "sliding_attention": attn_mask,
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return self._build_text_output(hidden_states, past_key_values)

    def _build_text_output(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None,
    ) -> BaseModelOutputWithPast:
        """Override to return family-specific output class."""
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class FamilyModel(Qwen3VLModel):
    """Base for family-specific ``Model`` subclasses.

    Subclasses must set ``config_class`` and optionally override
    ``_build_model_components`` to customize initialization.
    """

    config_class: type
    _no_split_modules: ClassVar[list[str]] = []

    def __init__(self, config: Any) -> None:
        Qwen3NextPreTrainedModel.__init__(self, config)
        self._build_model_components(config)
        self.rope_deltas = None
        self.post_init()

    def _build_model_components(self, config: Any) -> None:
        """Override to add family-specific model components."""

    def get_video_features(
        self, **super_kwargs: Any
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        return super().get_video_features(**super_kwargs)

    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Any, ...] | FamilyModelOutputWithPast:
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


class FamilyForCausalLM(Qwen3ForCausalLM):
    """Base for family-specific ``ForCausalLM`` subclasses."""

    config_class: type
    config: Any
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
        r"^model.visual.*",
    ]

    def __init__(self, config: Any) -> None:
        if hasattr(config, "text_config") and not hasattr(config, "vocab_size"):
            config = config.text_config
        super().__init__(config)
        self.model = self._build_causal_lm_model(config)

    def _build_causal_lm_model(self, config: Any) -> nn.Module:
        """Override to return family-specific text model."""

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""Compute total gate balance regularization loss across all GDN-2 mixer layers."""
        return self.model.get_gate_regularization_loss(target=target)

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
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
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
        """Override to apply family-specific post-processing to causal LM outputs."""


class FamilyForTokenClassification(
    GenericForTokenClassification, FamilyPreTrainedModel
):
    """Base for family-specific ``ForTokenClassification`` subclasses."""

    config_class: type
    config: Any


class FamilyForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Base for family-specific ``ForConditionalGeneration`` subclasses."""

    config_class: type
    config: Any
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
    ]

    def __init__(self, config: Any) -> None:
        FamilyPreTrainedModel.__init__(self, config)
        self.model = self._build_conditional_model(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    def _build_conditional_model(self, config: Any) -> nn.Module:
        """Override to return family-specific model."""

    def get_video_features(
        self, **super_kwargs: Any
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        return super().get_video_features(**super_kwargs)

    def get_image_features(
        self, **super_kwargs: Any
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | FamilyModelOutputWithPast:
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
        """Override to apply family-specific loss post-processing."""

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""Compute total gate balance regularization loss across all GDN-2 mixer layers."""
        return self.model.get_gate_regularization_loss(target=target)


class FamilyTextForSequenceClassification(
    GenericForSequenceClassification, FamilyPreTrainedModel
):
    """Base for family-specific ``TextForSequenceClassification`` subclasses."""

    config_class: type
    config: Any
    input_modalities: ClassVar[tuple[str, ...]] = ("text",)


class FamilyForSequenceClassification(
    GenericForSequenceClassification, FamilyPreTrainedModel
):
    """Base for family-specific ``ForSequenceClassification`` subclasses."""

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
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        return super().forward(
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
