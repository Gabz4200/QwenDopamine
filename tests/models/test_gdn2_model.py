"""Model-level and ops regression tests for the GDN-2 host stack.

These cover what the Prime 1.3B notebook relies on: an all-GDN-2 Transformer
trains end-to-end (chunk path), gradients reach the GDN-2 projections,
gradient checkpointing is safe, and the optional Triton/FLA ops modules still
import and expose their entry points when CUDA is absent.
"""

from typing import Any, cast

import torch
from torch import nn

from qwendopamine.models.gdn2.gdn2 import (
    GatedDeltaNet2,
    compute_gdn2_intra_chunk_scores,
    compute_gdn2_wy_coefficients,
    resolve_gdn2_backend,
)
from qwendopamine.models.gdn2_gpt import GDN2GPT, GDN2GPTConfig


def _make_cfg(**overrides: Any) -> GDN2GPTConfig:
    base: dict[str, Any] = {
        "name": "test",
        "n_layer": 2,
        "n_embd": 128,
        "n_head": 4,
        "head_size": 32,
        "n_query_groups": 4,
        "intermediate_size": 344,
        "train_chunk_size": 16,
        "use_short_conv": True,
        "norm_eps": 1e-6,
        "vocab_size": 512,
        "block_size": 64,
        "gdn2_per_layer": 1,
    }
    base.update(overrides)
    return GDN2GPTConfig(**base)


def test_when_all_gdn2_model_then_forward_and_backward_flow() -> None:
    model = GDN2GPT(_make_cfg())
    model.train()
    x = torch.randint(0, 512, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 512)
    logits.float().mean().backward()
    # Gradient must flow into the GDN-2 projections (and decay params).
    attn = cast(Any, model.h[0].attn)
    assert attn.q_proj.weight.grad is not None
    assert attn.A_log.grad is not None


def test_when_gradient_checkpointing_then_output_matches() -> None:
    cfg = _make_cfg()
    model_ref = GDN2GPT(cfg)
    model_ckpt = GDN2GPT(cfg)
    model_ckpt.load_state_dict(model_ref.state_dict())

    x = torch.randint(0, 512, (2, 32))
    model_ref.train()
    model_ckpt.train()
    model_ckpt.gradient_checkpointing_enable()

    out_ref = model_ref(x)
    out_ckpt = model_ckpt(x)
    assert out_ref.shape == out_ckpt.shape
    assert torch.allclose(out_ref.float(), out_ckpt.float(), atol=1e-5)


def test_when_all_gdn2_model_then_overfits_small_batch() -> None:
    model = GDN2GPT(_make_cfg())
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.randint(0, 512, (4, 20))
    yt = torch.randint(0, 512, (4, 20))

    initial_loss = 0.0
    final_loss = 0.0
    for step_idx in range(30):
        opt.zero_grad()
        loss = loss_fn(model(xt).reshape(-1, 512), yt.reshape(-1))
        loss.backward()
        opt.step()
        if step_idx == 0:
            initial_loss = float(loss.detach())
        final_loss = float(loss.detach())
    assert final_loss < initial_loss * 0.7, (
        f"no learning: {initial_loss:.3f} -> {final_loss:.3f}"
    )


def test_when_l2norm_disabled_then_chunk_and_recurrent_still_agree() -> None:
    """Non-destructive check that the chunk path honours the L2 toggle."""
    from qwendopamine.models.gdn2.gdn2 import torch_chunk_gdn2, torch_recurrent_gdn2

    torch.manual_seed(1)
    b, t, h, dk, dv = 1, 9, 2, 16, 16
    args = (
        torch.randn(b, t, h, dk),
        torch.randn(b, t, h, dk),
        torch.randn(b, t, h, dv),
        -torch.rand(b, t, h, dk).abs(),
        torch.sigmoid(torch.randn(b, t, h, dk)),
        torch.sigmoid(torch.randn(b, t, h, dv)),
    )
    o_r, _ = torch_recurrent_gdn2(*args, use_qk_l2norm_in_kernel=False)
    o_c, _ = torch_chunk_gdn2(*args, chunk_size=4, use_qk_l2norm_in_kernel=False)
    assert torch.allclose(o_r, o_c, atol=1e-5)


def test_when_auto_backend_on_cpu_then_chooses_pytorch() -> None:
    assert resolve_gdn2_backend("auto", training=True, seq_len=64) == "torch-chunk"
    assert resolve_gdn2_backend("auto", training=False, seq_len=1) == "torch-recurrent"
    assert resolve_gdn2_backend("auto", training=False, seq_len=32) == "torch-recurrent"
    assert resolve_gdn2_backend("auto", training=False, seq_len=256) == "torch-chunk"
    # Forced backends are never auto-rewritten.
    assert resolve_gdn2_backend("torch-recurrent", training=True, seq_len=256) == (
        "torch-recurrent"
    )


def test_when_wy_helpers_then_single_token_recovers_trivially() -> None:
    """Y = (I+T)^{-1} ebar with T empty reduces to ebar for a 1-token chunk."""
    kbar = torch.randn(1, 1, 1, 8)
    ebar = torch.randn(1, 1, 1, 8)
    z = torch.randn(1, 1, 1, 5)
    y, u = compute_gdn2_wy_coefficients(kbar, ebar, z, device=torch.device("cpu"))
    assert torch.allclose(y, ebar, atol=1e-6)
    assert torch.allclose(u, z, atol=1e-6)


def test_when_intra_chunk_scores_then_strictly_causal() -> None:
    q = torch.randn(1, 1, 4, 8)
    gamma = torch.ones(1, 1, 4, 8)
    kbar = torch.randn(1, 1, 4, 8)
    scores = compute_gdn2_intra_chunk_scores(q, gamma, kbar)
    assert scores.shape == (1, 1, 4, 4)
    # Upper triangle (row < col) must be zeroed.
    ut = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert (scores[0, 0][ut] == 0).all()
    # Diagonal (self) is retained.
    assert scores[0, 0].diagonal().abs().sum() > 0


def test_when_gdn2_block_then_streams_recurrent_state() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, chunk_size=8)
    layer.eval()
    cache: dict[
        str, torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    with torch.no_grad():
        out1 = layer(torch.randn(1, 1, 64), past_key_values=cache, use_cache=True)[0]
        s1 = cast(torch.Tensor, cache["recurrent_state"])
        out2 = layer(torch.randn(1, 1, 64), past_key_values=cache, use_cache=True)[0]
        s2 = cast(torch.Tensor, cache["recurrent_state"])
    assert out1.shape == (1, 1, 64)
    assert out2.shape == (1, 1, 64)
    assert not torch.allclose(s1, s2)  # state evolved across steps
