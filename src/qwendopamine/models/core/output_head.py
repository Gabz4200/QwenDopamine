r"""Output language modeling prediction head layers."""

from __future__ import annotations

import torch
from torch import nn


class LMHead(nn.Module):
    r"""LMHead(hidden_size: int, vocab_size: int) -> None

    Vocabulary projection head with an intermediate SiLU-activated dense
    layer and LayerNorm, as used in Qwen3.5.

    Args:
        hidden_size (int): Input and hidden dimension of the dense layer.
        vocab_size (int): Output vocabulary size.

    Examples::
        >>> head = LMHead(hidden_size=512, vocab_size=1000)
        >>> logits = head(torch.randn(2, 4, 512))
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""forward(hidden_states: torch.Tensor) -> torch.Tensor

        Project hidden states to vocabulary logits.

        Args:
            hidden_states (torch.Tensor): Input tensor ``[..., hidden_size]``.

        Returns:
            torch.Tensor: Logits ``[..., vocab_size]``.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.functional.silu(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.decoder(hidden_states)
