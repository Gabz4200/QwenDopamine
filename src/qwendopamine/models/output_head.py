r"""Output language modeling prediction head layers."""

from __future__ import annotations

import torch
from torch import nn


class LMHead(nn.Module):
    r"""Dense-SiLU-LayerNorm vocabulary projection head."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Project hidden states to vocabulary logits."""
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.functional.silu(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.decoder(hidden_states)
