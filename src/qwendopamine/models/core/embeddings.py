r"""Token and position embedding layers for causal language models."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbeddings(nn.Module):
    r"""TokenEmbeddings(vocab_size: int, hidden_size: int) -> None

    Dense token embedding lookup table.

    Args:
        vocab_size (int): Number of tokens in the vocabulary.
        hidden_size (int): Embedding dimension.

    Examples::
        >>> emb = TokenEmbeddings(vocab_size=1000, hidden_size=512)
        >>> out = emb(torch.randint(0, 1000, (2, 4)))
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        r"""Embed token ids.

        Args:
            input_ids (torch.Tensor): Token indices ``[B, T]``.

        Returns:
            torch.Tensor: Embedded representations ``[B, T, hidden_size]``.
        """
        return self.weight[input_ids]


class PositionEmbeddings(nn.Module):
    r"""PositionEmbeddings(max_position_embeddings: int, hidden_size: int) -> None

    Learned absolute position embedding lookup table.

    Args:
        max_position_embeddings (int): Maximum sequence length supported.
        hidden_size (int): Embedding dimension.

    Examples::
        >>> pe = PositionEmbeddings(max_position_embeddings=2048, hidden_size=512)
        >>> pos = pe(torch.arange(4))
    """

    def __init__(self, max_position_embeddings: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_position_embeddings, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        r"""Embed position ids.

        Args:
            position_ids (torch.Tensor): Position indices ``[B, T]``.

        Returns:
            torch.Tensor: Position embeddings ``[B, T, hidden_size]``.
        """
        return self.weight[position_ids]
