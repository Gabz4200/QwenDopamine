"""HF nn.Module block wrapper around GatedDeltaNet2.

Provides :class:`GDN2HFBlock`, which adapts a HF PreTrainedConfig (or a plain
GDN2Config / dict) into a runnable token-mixing layer compatible with HF's
``transformers`` AutoModel registration.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwendopamine.integrations.huggingface.configs import (
    GDN2HFConfig,
    PreTrainedConfig,
)


class GDN2HFBlock(nn.Module):
    r"""GDN2HFBlock(config: PreTrainedConfig | Any, layer_idx: int | None = None, **kwargs: Any) -> None

    Hugging Face compatible nn.Module block wrapper around GatedDeltaNet2.

    Args:
        config (PreTrainedConfig | Any): HF config or GDN2Config/dict.
        layer_idx (int | None): Layer index for cache naming. Default: ``None``.
        **kwargs: Extra kwargs forwarded to GatedDeltaNet2.
    """

    def __init__(
        self,
        config: PreTrainedConfig | Any,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from qwendopamine.models.gdn2 import GatedDeltaNet2

        if isinstance(config, GDN2HFConfig):
            self.config: GDN2HFConfig = config
        else:
            self.config = GDN2HFConfig.from_gdn2_config(config, **kwargs)
        self.layer_idx = layer_idx
        self.mixer = GatedDeltaNet2(self.config, layer_idx=layer_idx, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
        r"""Forward pass delegating to the underlying GatedDeltaNet2 mixer.

        Args:
            hidden_states (torch.Tensor): Input ``[B, T, D]``.
            attention_mask (torch.Tensor | None): Padding mask ``[B, T]``.
            past_key_values (Any): Cache for decoding.
            use_cache (bool | None): Return updated cache. Default: ``False``.
            output_attentions (bool | None): Whether to return attentions.
            **kwargs: Extra kwargs forwarded to the mixer.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, Any]:
            ``(hidden_states, attentions, past_key_values)``.
        """
        return self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )


__all__ = ["GDN2HFBlock"]
