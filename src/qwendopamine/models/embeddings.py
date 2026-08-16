r"""Token and position embedding layers for causal language models."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbeddings(nn.Module):
    r"""TokenEmbeddings(vocab_size, hidden_size)

    Constructs a standard dense token embedding table initialized with zero-mean normal distribution.

    Args:
        vocab_size (int): Total vocabulary size (number of discrete tokens).
        hidden_size (int): Dimension of output token embedding vectors.

    Examples::

        >>> emb = TokenEmbeddings(vocab_size=1000, hidden_size=64)
        >>> input_ids = torch.tensor([[1, 2, 4], [5, 6, 7]])
        >>> out = emb(input_ids)
        >>> out.shape
        torch.Size([2, 3, 64])
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        r"""forward(input_ids) -> Tensor

        Args:
            input_ids (Tensor): Long tensor of token indices of shape :math:`(B, L)`.

        Returns:
            Tensor: Embedded sequence feature tensor of shape :math:`(B, L, \text{hidden\_size})`.
        """
        return self.weight[input_ids]


class PositionEmbeddings(nn.Module):
    r"""PositionEmbeddings(max_position_embeddings, hidden_size)

    Constructs learned absolute position embedding lookup table.

    Args:
        max_position_embeddings (int): Maximum sequence length supported by the position table.
        hidden_size (int): Dimension of output positional vectors.

    Examples::

        >>> pos_emb = PositionEmbeddings(max_position_embeddings=512, hidden_size=64)
        >>> pos_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
        >>> out = pos_emb(pos_ids)
        >>> out.shape
        torch.Size([2, 3, 64])
    """

    def __init__(self, max_position_embeddings: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_position_embeddings, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        r"""forward(position_ids) -> Tensor

        Args:
            position_ids (Tensor): Long tensor of position indices of shape :math:`(B, L)`.

        Returns:
            Tensor: Position embedding vectors of shape :math:`(B, L, \text{hidden\_size})`.
        """
        return self.weight[position_ids]
