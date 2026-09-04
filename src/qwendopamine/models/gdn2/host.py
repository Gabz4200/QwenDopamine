# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

"""Lightweight decoder-only host stacking :class:`GatedDeltaNet2` blocks.

This is a minimal GPT-style host (embedding -> GDN-2 blocks -> final norm ->
LM head) used by the model initialization tests. The production hybrid model
is :class:`qwendopamine.models.gdn2_gpt.GDN2GPT`, which mixes standard
attention, MLP, and GDN-2 layers.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.models.core.normalization import RMSNorm
from qwendopamine.models.gdn2 import GatedDeltaNet2
from qwendopamine.models.gdn2.config import GDN2Config


class Block(nn.Module):
    r"""Transformer block with GatedDeltaNet2 token mixer.

    Performs the standard decoder operation: norm -> attention (GDN-2 token
    mixer) -> residual connection. The GDN-2 attention replaces standard
    scaled-dot-product attention with a channel-wise gated recurrent state
    mixer.

    Attributes:
        config: The :class:`GDN2Config` controlling hidden size, number of
            heads, convolution kernel size, and backend selection.
        layer_idx: Index of this block in the overall model.
        norm: ``RMSNorm`` applied to the input before the attention.
        attn: The :class:`GatedDeltaNet2` token-mixing layer.
    """

    def __init__(self, config: GDN2Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedDeltaNet2(config, layer_idx=layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any | None]:
        r"""Forward pass of the GDN-2 block.

        Args:
            x: Hidden-state tensor ``[B, T, D]``.
            past_key_values: Optional ``Cache`` or dict with past state.
            use_cache: If ``True``, return past_key_values.

        Returns:
            A tuple ``(x + attn(x), past_key_values)``.
        """
        residual = x
        x = self.norm(x)
        out, _, past_key_values = self.attn(
            x, past_key_values=past_key_values, use_cache=use_cache
        )
        return residual + out, past_key_values


class GDN2Host(nn.Module):
    r"""GPT-style decoder using only GDN-2 blocks.

    A minimal decoder (embedding -> GDN-2 blocks -> final norm -> LM head) used
    by the model initialization tests. Each ``Block`` replaces standard
    attention with the GDN-2 token mixer.

    The host handles token embedding, stacking ``Block`` s according to
    ``config.num_layers``, final ``RMSNorm`` and ``lm_head`` projection to
    vocabulary logits.

    Shapes:
        input_ids : ``[B, T]`` — token indices.
        output : ``[B, T, hidden_size]`` — logits over the vocabulary.
    """

    def __init__(self, config: GDN2Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Block(config, i) for i in range(getattr(config, "num_layers", 24))]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        from ._init_guard import is_already_initialised, mark_initialised

        if is_already_initialised(module):
            return
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            module.reset_parameters()
        mark_initialised(module)

    def forward(
        self, input_ids: torch.Tensor, past_key_values: Any | None = None
    ) -> tuple[torch.Tensor, Any | None]:
        r"""Forward pass of the GDN-2 decoder host.

        Args:
            input_ids: Token indices ``[B, T]``.
            past_key_values: Optional ``Cache`` for autoregressive decoding.

        Returns:
            ``(logits, past_key_values)`` where logits are ``[B, T, vocab_size]``.
        """
        x = self.embed(input_ids)
        for layer in self.layers:
            x, past_key_values = layer(x, past_key_values=past_key_values)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, past_key_values


__all__ = ["Block", "GDN2Host"]
