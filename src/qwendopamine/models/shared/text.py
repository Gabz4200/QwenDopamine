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
"""Family-specific ``TextModel`` base class."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GenericForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextModel

from qwendopamine.models._transformers_utils import expand_position_ids_to_multimodal
from qwendopamine.models.shared.pretrained import FamilyPreTrainedModel


class FamilyTextModel(Qwen3NextModel):
    r"""Base for family-specific ``TextModel`` subclasses.

    Subclasses must set ``config_class`` and optionally override
    ``_build_text_layers`` to customize layer initialization.
    """

    config_class: type
    config: Any

    def __init__(self, config: Any) -> None:
        r"""__init__(self, config: Any) -> None

        FamilyTextModel(config: Any) -> None

                Initialize the text tower and build family-specific layers.

                Args:
                    config (Any): Text model configuration.
        """
        super().__init__(config)
        self._build_text_layers(config)

    def _build_text_layers(self, config: Any) -> None:
        r"""_build_text_layers(self, config: Any) -> None

        Override to add family-specific text layers after parent init.

        Args:
            self - .
            config (Any) - .
        """

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        r"""forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, **kwargs) -> BaseModelOutputWithPast

        Run the text transformer and normalize outputs.

        Args:
            input_ids (torch.LongTensor | None): Token indices.
            attention_mask (torch.Tensor | None): Attention mask.
            position_ids (torch.LongTensor | None): Position indices.
            past_key_values (Cache | None): Cached key/value states.
            inputs_embeds (torch.FloatTensor | None): Precomputed embeddings.
            use_cache (bool | None): Whether to return past key/values.
            **kwargs (Any): Extra model inputs.

        Returns:
            BaseModelOutputWithPast: Last hidden state and optional cache.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            if past_key_values is not None and hasattr(
                past_key_values, "get_seq_length"
            ):
                # ``DynamicCache`` (the common case) supports
                # ``get_seq_length``; a custom cache without that
                # method is treated as having zero past tokens.
                try:
                    past_seen_tokens = past_key_values.get_seq_length()
                except (ValueError, NotImplementedError):
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
        return self._build_text_output(hidden_states, past_key_values)

    def _build_text_output(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None,
    ) -> BaseModelOutputWithPast:
        r"""_build_text_output(self, hidden_states: torch.Tensor, past_key_values: Cache | None) -> BaseModelOutputWithPast

        Override to return family-specific output class.

        Args:
            self - .
            hidden_states (torch.Tensor) - .
            past_key_values (Cache | None) - .

        Returns:
            BaseModelOutputWithPast - .
        """
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class FamilyTextForSequenceClassification(
    GenericForSequenceClassification, FamilyPreTrainedModel
):
    r"""Base for family-specific ``TextForSequenceClassification`` subclasses."""

    config_class: type
    config: Any
    input_modalities: ClassVar[tuple[str, ...]] = ("text",)
