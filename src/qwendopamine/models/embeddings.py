r"""Token and position embedding layers for causal language models."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbeddings(nn.Module):
    r"""Dense token embedding table."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        r"""Embed token ids."""
        return self.weight[input_ids]


class PositionEmbeddings(nn.Module):
    r"""Learned absolute position embedding lookup table."""

    def __init__(self, max_position_embeddings: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_position_embeddings, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        r"""Embed position ids."""
        return self.weight[position_ids]
