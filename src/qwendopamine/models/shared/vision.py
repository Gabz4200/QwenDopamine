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
"""Family-specific ``VisionModel`` base class."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


class FamilyVisionModel(Qwen3VLVisionModel):
    r"""Base for family-specific ``VisionModel`` subclasses.

    Subclasses only need to set ``config_class``; the ``__init__`` and
    ``forward`` implementations are shared.
    """

    config_class: type
    config: Any
    _no_split_modules: ClassVar[list[str]] = []

    def __init__(self, config: Any, *inputs: Any, **kwargs: Any) -> None:
        r"""__init__(self, config: Any, *inputs: Any, **kwargs: Any) -> None

        FamilyVisionModel(config: Any, *inputs: Any, **kwargs: Any) -> None

                Initialize the vision tower and remove family-specific attributes.

                Args:
                    config (Any): Vision model configuration.
                    *inputs (Any): Extra positional arguments forwarded to the parent.
                    **kwargs (Any): Extra keyword arguments forwarded to the parent.
        """
        super().__init__(config, *inputs, **kwargs)
        # We deliberately do NOT call ``self.post_init()`` here.
        # Upstream ``Qwen3VLVisionModel`` skips it too: the vision
        # module has no weight-initialisation hooks of its own, and
        # any custom subclass that *does* need ``post_init`` should
        # call it explicitly so the intent is visible at the call
        # site. ``FamilyModel.__init__`` (the text+vision parent)
        # *does* call ``post_init``; that one runs once on the
        # full multimodal model.
        self._delete_vision_attributes()

    def _delete_vision_attributes(self) -> None:
        r"""_delete_vision_attributes(self) -> None

        Override to delete family-specific vision attributes after init.

        Args:
            self - .
        """

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        r"""forward(hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any) -> BaseModelOutputWithPooling

        Run the vision backbone and return pooled patch embeddings.

        Args:
            hidden_states (torch.Tensor): Patch-embedded image tokens.
            grid_thw (torch.Tensor): Spatial grid layout ``[T, H, W]``.
            **kwargs (Any): Forwarded to internal vision operations.

        Returns:
            BaseModelOutputWithPooling: Last hidden state and pooled output.
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
        # Patch-embed + interpolated positional-embedding add. The
        # ``interp_weights`` reduce a small per-patch weight tensor
        # (one weight per side of the grid) into a single position
        # embedding vector.
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(
            1
        )
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        # Build per-token RoPE position embeddings and split into
        # (cos, sin) halves consumed by every attention block.
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
