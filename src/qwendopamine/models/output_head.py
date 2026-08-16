r"""Output language modeling prediction head layers."""

from __future__ import annotations

import torch
from torch import nn


class LMHead(nn.Module):
    r"""LMHead(hidden_size, vocab_size)

    Projects final hidden states through a dense linear layer, SiLU activation, LayerNorm, and vocabulary projection.

    .. math::
        \text{Logits} = W_{\text{vocab}} \cdot \text{LayerNorm}(\text{SiLU}(W_{\text{dense}} \cdot x))

    Args:
        hidden_size (int): Hidden dimension size of model feature representation.
        vocab_size (int): Vocabulary size for final output logit generation.

    Examples::

        >>> head = LMHead(hidden_size=64, vocab_size=1000)
        >>> hidden_states = torch.randn(2, 5, 64)
        >>> logits = head(hidden_states)
        >>> logits.shape
        torch.Size([2, 5, 1000])
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""forward(hidden_states) -> Tensor

        Args:
            hidden_states (Tensor): Final hidden state sequence tensor of shape :math:`(B, L, \text{hidden\_size})`.

        Returns:
            Tensor: Vocabulary logit predictions of shape :math:`(B, L, \text{vocab\_size})`.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.functional.silu(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.decoder(hidden_states)
