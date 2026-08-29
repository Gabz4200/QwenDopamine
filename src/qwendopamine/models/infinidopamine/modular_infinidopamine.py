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
"""PyTorch InfiniDopamine model."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

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
    accepts_precomputed_kwargs,
    can_return_tuple,
    capture_outputs,
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    init,
    merge_with_config_defaults,
    unwrap_gated_delta_rule_fns,
)
from qwendopamine.models.infinidopamine.configs import (
    InfiniDopamineConfig,
    InfiniDopamineTextConfig,
    InfiniDopamineVisionConfig,
)
from qwendopamine.models.infinidopamine.decoder_layer import (
    InfiniDopamineAttention,
    InfiniDopamineDecoderLayer,
    InfiniDopamineGatedDeltaNet,
    InfiniDopamineGatedRewardNet,  # noqa: F401 — re-exported for package API
    )
from qwendopamine.models.infinidopamine.rotary_embeddings import (
    InfiniDopamineTextRotaryEmbedding,
    InfiniDopamineVisionRotaryEmbedding,
)

unwrap_gated_delta_rule_fns()



class InfiniDopaminePreTrainedModel(Qwen3NextPreTrainedModel):
    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
    _no_split_modules: ClassVar[list[str]] = [
        "InfiniDopamineDecoderLayer",
        "InfiniDopamineVisionBlock",
    ]
    _can_record_outputs: ClassVar[dict[str, type]] = {
        "hidden_states": InfiniDopamineDecoderLayer,
        "attentions": InfiniDopamineAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, InfiniDopamineGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device)
                .uniform_(0.01, 16)
                .log_(),
            )
            init.zeros_(module.betas)
            init.zeros_(module.in_proj_gate.weight)
        elif isinstance(module, InfiniDopamineVisionRotaryEmbedding):
            inv_freq = 1.0 / (
                module.theta
                ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim)
            )
            init.copy_(module.inv_freq, inv_freq)


class InfiniDopamineVisionModel(Qwen3VLVisionModel):
    config_class = InfiniDopamineVisionConfig
    config: InfiniDopamineVisionConfig
    _no_split_modules: ClassVar[list[str]] = ["InfiniDopamineVisionBlock"]

    def __init__(self, config: Any, *inputs: Any, **kwargs: Any) -> None:
        super().__init__(config, *inputs, **kwargs)
        del self.deepstack_visual_indexes
        del self.deepstack_merger_list

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


class InfiniDopamineModelOutputWithPast(Qwen3VLModelOutputWithPast):
    pass


class InfiniDopamineTextModel(Qwen3NextModel):
    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig

    def __init__(self, config: InfiniDopamineTextConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                InfiniDopamineDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = InfiniDopamineTextRotaryEmbedding(config=config)

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
            position_ids = (
                torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                )
                + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(
                4, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[:, None, :].expand(
                4, position_ids.shape[0], -1
            )

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            attn_mask = (
                create_sliding_window_causal_mask(**mask_kwargs)
                if getattr(self.config, "sliding_window", None) is not None
                else create_causal_mask(**mask_kwargs)
            )
            causal_mask_mapping = {
                "full_attention": attn_mask,
                "sliding_attention": attn_mask,
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        reward_values = kwargs.pop("reward_values", None)
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                reward_values=reward_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return InfiniDopamineModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""Compute total gate balance regularization loss across all GDN-2 mixer layers."""
        losses: list[torch.Tensor] = []
        for layer in self.layers[: self.config.num_hidden_layers]:
            if hasattr(layer, "linear_attn") and hasattr(
                layer.linear_attn, "get_gate_regularization_loss"
            ):
                losses.append(
                    layer.linear_attn.get_gate_regularization_loss(target=target)
                )
        if not losses:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def load_qwen35_weights(
        self,
        weights: dict[str, torch.Tensor] | nn.Module,
        strict: bool = True,
    ) -> Any:
        r"""Load pretrained Qwen3.5 (GDN-1) weights into InfiniDopamine (GDN-2 with SWA)."""
        if isinstance(weights, nn.Module):
            state_dict = weights.state_dict()
        else:
            state_dict = dict(weights)

        has_full_prefix = any(
            k.startswith(("model.language_model.", "model.", "language_model."))
            for k in state_dict
        )
        if has_full_prefix:
            remapped_state_dict: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith("model.language_model."):
                    new_k = new_k[len("model.language_model.") :]
                elif new_k.startswith("language_model."):
                    new_k = new_k[len("language_model.") :]
                elif new_k.startswith("model."):
                    new_k = new_k[len("model.") :]
                if not (
                    k.startswith(("model.visual.", "visual.", "mtp."))
                    or k == "lm_head.weight"
                ):
                    remapped_state_dict[new_k] = v
            state_dict = remapped_state_dict

        return self.load_state_dict(state_dict, strict=strict)


class InfiniDopamineModel(Qwen3VLModel):
    config_class = InfiniDopamineConfig
    _no_split_modules: ClassVar[list[str]] = [
        "InfiniDopamineDecoderLayer",
        "InfiniDopamineVisionBlock",
    ]

    def __init__(self, config: InfiniDopamineConfig) -> None:
        InfiniDopaminePreTrainedModel.__init__(self, config)
        self.visual = InfiniDopamineVisionModel(config.vision_config)
        self.language_model = InfiniDopamineTextModel(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_video_features(
        self, **super_kwargs: Any
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        return super().get_video_features(**super_kwargs)

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Any, ...] | BaseModelOutputWithPooling:
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output: BaseModelOutputWithPooling = self.visual(
            pixel_values, image_grid_thw=image_grid_thw, return_dict=True, **kwargs
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
    ) -> tuple[Any, ...] | InfiniDopamineModelOutputWithPast:
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
        return InfiniDopamineModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class InfiniDopamineForCausalLM(Qwen3ForCausalLM):
    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
        r"^model.visual.*",
    ]

    def __init__(self, config: Any) -> None:
        if hasattr(config, "text_config") and not hasattr(config, "vocab_size"):
            config = config.text_config
        super().__init__(config)
        self.model = InfiniDopamineTextModel(config)

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
        if (
            labels is not None
            and getattr(outputs, "loss", None) is not None
            and self.training
            and getattr(self.config, "gate_loss_weight", 0.0) > 0.0
        ):
            target = getattr(self.config, "gate_target_balance", 0.5)
            gate_loss = self.get_gate_regularization_loss(target=target)
            outputs.loss = outputs.loss + self.config.gate_loss_weight * gate_loss
        return outputs

    def load_qwen35_weights(
        self,
        weights: dict[str, torch.Tensor] | nn.Module,
        strict: bool = True,
    ) -> Any:
        r"""Load pretrained Qwen3.5 (GDN-1) weights into InfiniDopamine (GDN-2 with SWA)."""
        if isinstance(weights, nn.Module):
            state_dict = weights.state_dict()
        else:
            state_dict = dict(weights)

        has_language_model_prefix = any(
            k.startswith(("model.language_model.", "language_model."))
            for k in state_dict
        )
        if has_language_model_prefix:
            remapped_state_dict: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.startswith("model.language_model."):
                    remapped_state_dict[k.replace("model.language_model.", "model.")] = v
                elif k.startswith("language_model."):
                    remapped_state_dict[k.replace("language_model.", "model.")] = v
                elif k == "lm_head.weight" or not k.startswith(("model.visual.", "visual.", "mtp.")):
                    remapped_state_dict[k] = v
            state_dict = remapped_state_dict

        return self.load_state_dict(state_dict, strict=strict)


class InfiniDopamineForTokenClassification(
    GenericForTokenClassification, InfiniDopaminePreTrainedModel
):
    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig


class InfiniDopamineForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
    ]

    def __init__(self, config: InfiniDopamineConfig) -> None:
        InfiniDopaminePreTrainedModel.__init__(self, config)
        self.model = InfiniDopamineModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

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
    ) -> tuple | InfiniDopamineModelOutputWithPast:
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

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        if (
            loss is not None
            and self.training
            and getattr(self.config, "gate_loss_weight", 0.0) > 0.0
        ):
            target = getattr(self.config, "gate_target_balance", 0.5)
            gate_loss = self.model.get_gate_regularization_loss(target=target)
            loss = loss + self.config.gate_loss_weight * gate_loss

        return InfiniDopamineModelOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""Compute total gate balance regularization loss across all GDN-2 mixer layers."""
        return self.model.get_gate_regularization_loss(target=target)

    def load_qwen35_weights(
        self,
        weights: dict[str, torch.Tensor] | nn.Module,
        strict: bool = True,
    ) -> Any:
        r"""Load pretrained Qwen3.5 (GDN-1) weights into InfiniDopamine (GDN-2 with SWA)."""
        if isinstance(weights, nn.Module):
            state_dict = weights.state_dict()
        else:
            state_dict = dict(weights)

        vision_state: dict[str, torch.Tensor] = {}
        text_state: dict[str, torch.Tensor] = {}
        lm_head_state: dict[str, torch.Tensor] = {}

        for k, v in state_dict.items():
            if k == "lm_head.weight":
                lm_head_state[k] = v
            elif k.startswith("model.visual."):
                vision_state[k[len("model.visual.") :]] = v
            elif k.startswith("model.language_model."):
                text_state[k[len("model.") :]] = v
            elif k.startswith("language_model."):
                text_state[k] = v
            elif k.startswith("visual."):
                vision_state[k[len("visual.") :]] = v
            elif k.startswith("mtp."):
                continue
            elif strict:
                text_state[k] = v

        load_info: list[str] = []
        all_missing: list[str] = []
        all_unexpected: list[str] = []

        if vision_state:
            missing_v, unexpected_v = self.model.visual.load_state_dict(
                vision_state, strict=strict
            )
            all_missing.extend(missing_v)
            all_unexpected.extend(unexpected_v)
            load_info.append(
                f"vision: loaded {len(vision_state) - len(missing_v)} keys "
                f"({len(missing_v)} missing, {len(unexpected_v)} unexpected)"
            )

        if text_state:
            missing_t, unexpected_t = self.model.language_model.load_qwen35_weights(
                text_state, strict=strict
            )
            all_missing.extend(missing_t)
            all_unexpected.extend(unexpected_t)
            load_info.append(
                f"text: loaded {len(text_state) - len(missing_t)} keys "
                f"({len(missing_t)} missing, {len(unexpected_t)} unexpected)"
            )

        if lm_head_state:
            self.lm_head.weight.data.copy_(lm_head_state["lm_head.weight"])
            load_info.append("lm_head: loaded 1 key")

        return _IncompatibleKeys(all_missing, all_unexpected)


class InfiniDopamineTextForSequenceClassification(
    GenericForSequenceClassification, InfiniDopaminePreTrainedModel
):
    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig
    input_modalities = ("text",)


class InfiniDopamineForSequenceClassification(
    GenericForSequenceClassification, InfiniDopaminePreTrainedModel
):
    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
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
    "InfiniDopamineConfig",
    "InfiniDopamineDecoderLayer",
    "InfiniDopamineForCausalLM",
    "InfiniDopamineForConditionalGeneration",
    "InfiniDopamineForSequenceClassification",
    "InfiniDopamineForTokenClassification",
    "InfiniDopamineGatedDeltaNet",
    "InfiniDopamineModel",
    "InfiniDopaminePreTrainedModel",
    "InfiniDopamineTextConfig",
    "InfiniDopamineTextForSequenceClassification",
    "InfiniDopamineTextModel",
    "InfiniDopamineTextRotaryEmbedding",
    "InfiniDopamineVisionConfig",
    "InfiniDopamineVisionModel",
    "InfiniDopamineVisionRotaryEmbedding",
]
