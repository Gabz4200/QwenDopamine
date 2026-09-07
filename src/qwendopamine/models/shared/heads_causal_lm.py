"""Causal LM head for Qwen-family models."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.utils import can_return_tuple


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

    @staticmethod
    def _apply_causal_lm_postprocessing(
        outputs: CausalLMOutputWithPast,
    ) -> None:
        r"""_apply_causal_lm_postprocessing(self, outputs: CausalLMOutputWithPast) -> None

        Override to apply family-specific post-processing to causal LM outputs.

        Args:
            self - .
            outputs (CausalLMOutputWithPast) - .
        """

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