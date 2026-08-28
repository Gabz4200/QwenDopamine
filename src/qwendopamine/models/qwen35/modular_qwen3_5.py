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

from typing import Any, ClassVar

import torch
from torch import nn

from qwendopamine.models._hf_compat import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    Cache,
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
    accepts_precomputed_kwargs,
    can_return_tuple,
    capture_outputs,
    causal_conv1d_fn,
    causal_conv1d_update,
    create_causal_mask,
    create_recurrent_attention_mask,
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    init,
    logging,
    merge_with_config_defaults,
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


@use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")
@use_kernelized_func(
    [
        torch_recurrent_gated_delta_rule,
        torch_chunk_gated_delta_rule,
        causal_conv1d_fn,
        causal_conv1d_update,
    ]
)





class Qwen3_5PreTrainedModel(Qwen3NextPreTrainedModel):
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

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
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


class Qwen3_5VisionModel(Qwen3VLVisionModel):
    config_class = Qwen3_5VisionConfig
    config: Qwen3_5VisionConfig
    _no_split_modules: ClassVar[list[str]] = ["Qwen3_5VisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        del self.deepstack_visual_indexes
        del self.deepstack_merger_list

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
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
        pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(
            1
        )
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


class Qwen3_5ModelOutputWithPast(Qwen3VLModelOutputWithPast):
    pass


class Qwen3_5TextModel(Qwen3NextModel):
    config_class = Qwen3_5TextConfig
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

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

        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(
                4, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
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

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3_5Model(Qwen3VLModel):
    config_class = Qwen3_5Config
    _no_split_modules: ClassVar[list[str]] = [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
    ]

    def __init__(self, config: Qwen3_5Config) -> None:
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_video_features(self, **super_kwargs) -> tuple | BaseModelOutputWithPooling:
        # Same implementation as for images
        return super().get_video_features(**super_kwargs)

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
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
        input_ids: torch.LongTensor = None,
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
    ) -> tuple | Qwen3_5ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return Qwen3_5ModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class Qwen3_5ForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3_5TextConfig
    config: Qwen3_5TextConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
        r"^model.visual.*",
    ]

    def __init__(self, config: Any):
        if hasattr(config, "text_config") and not hasattr(config, "vocab_size"):
            config = config.text_config
        super().__init__(config)
        self.model = Qwen3_5TextModel(config)


class Qwen3_5ForTokenClassification(
    GenericForTokenClassification, Qwen3_5PreTrainedModel
):
    config_class = Qwen3_5Config
    config: Qwen3_5Config


class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config_class = Qwen3_5Config
    config: Qwen3_5Config
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
    ]

    def __init__(self, config: Qwen3_5Config) -> None:
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    def get_video_features(self, **super_kwargs) -> tuple | BaseModelOutputWithPooling:
        return super().get_video_features(**super_kwargs)

    def get_image_features(self, **super_kwargs) -> tuple | BaseModelOutputWithPooling:
        return super().get_image_features(**super_kwargs)


class Qwen3_5TextForSequenceClassification(
    GenericForSequenceClassification, Qwen3_5PreTrainedModel
):
    config_class = Qwen3_5TextConfig
    config: Qwen3_5TextConfig
    input_modalities = ("text",)


class Qwen3_5ForSequenceClassification(
    GenericForSequenceClassification, Qwen3_5PreTrainedModel
):
    config_class = Qwen3_5Config
    config: Qwen3_5Config
    def forward(
        self,
        input_ids: torch.LongTensor = None,
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
