"""Behavioral tests for embedding layers and output heads."""

from __future__ import annotations

import torch

from qwendopamine.models.embeddings import PositionEmbeddings, TokenEmbeddings
from qwendopamine.models.output_head import LMHead


def test_when_token_embeddings_forward_then_preserves_batch_seq_and_embeds() -> None:
    vocab_size, hidden_size = 100, 32
    embedding = TokenEmbeddings(vocab_size=vocab_size, hidden_size=hidden_size)
    input_ids = torch.tensor([[0, 5, 99], [1, 2, 3]], dtype=torch.long)

    out = embedding(input_ids)

    assert out.shape == (2, 3, hidden_size)
    assert not torch.isnan(out).any()


def test_when_token_embeddings_backward_then_gradients_flow() -> None:
    embedding = TokenEmbeddings(vocab_size=50, hidden_size=16)
    input_ids = torch.tensor([[2, 4, 6]], dtype=torch.long)
    out = embedding(input_ids)
    loss = out.sum()
    loss.backward()

    assert embedding.weight.grad is not None
    assert torch.any(embedding.weight.grad != 0.0)


def test_when_position_embeddings_forward_then_returns_positioned_tensors() -> None:
    max_positions, hidden_size = 64, 32
    pos_embedding = PositionEmbeddings(
        max_position_embeddings=max_positions, hidden_size=hidden_size
    )
    position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    out = pos_embedding(position_ids)

    assert out.shape == (1, 4, hidden_size)
    assert not torch.isnan(out).any()


def test_when_lm_head_forward_then_projects_hidden_to_vocab_logits() -> None:
    hidden_size, vocab_size = 32, 100
    head = LMHead(hidden_size=hidden_size, vocab_size=vocab_size)
    hidden_states = torch.randn(2, 4, hidden_size)

    logits = head(hidden_states)

    assert logits.shape == (2, 4, vocab_size)
    assert not torch.isnan(logits).any()


def test_when_lm_head_backward_then_dense_and_decoder_receive_gradients() -> None:
    head = LMHead(hidden_size=16, vocab_size=50)
    hidden_states = torch.randn(2, 4, 16, requires_grad=True)

    logits = head(hidden_states)
    loss = logits.sum()
    loss.backward()

    assert head.dense.weight.grad is not None
    assert head.decoder.weight.grad is not None
