"""Behavioral and shape tests for hardware-agnostic GatedDeltaNet2 (GDN-2)."""

from typing import Any, cast

import pytest
import torch
from transformers.cache_utils import DynamicCache

from qwendopamine.models.blocks.registry import BLOCKS, build_block
from qwendopamine.models.gdn2 import (
    GatedDeltaNet2,
    torch_chunk_gdn2,
    torch_recurrent_gdn2,
)
from qwendopamine.models.gdn2.config import GDN2Config


class DummyConfig:
    r"""DummyConfig: minimal GDN-2 config for tests."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        head_dim: int = 32,
        conv_size: int = 4,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_head = num_heads
        self.head_size = head_dim
        self.conv_size = conv_size
        self.norm_eps = rms_norm_eps
        self.rms_norm_eps = rms_norm_eps
        self.allow_neg_eigval = False
        self.expand_v = 1.0


def test_when_gdn2_forward_cpu_then_preserves_shape() -> None:
    layer = GatedDeltaNet2(hidden_size=128, num_heads=4, head_dim=32)
    x = torch.randn(2, 16, 128, dtype=torch.float32)
    out, attn, _cache = layer(x)
    assert out.shape == (2, 16, 128)
    assert attn is None


def test_when_gdn2_initialized_with_config_forward_then_preserves_shape() -> None:
    cfg = DummyConfig(hidden_size=128, num_heads=4, head_dim=32)
    layer = GatedDeltaNet2(cfg, layer_idx=0)
    x = torch.randn(2, 8, 128, dtype=torch.float32)
    out, _attn, _cache = layer(x)
    assert out.shape == (2, 8, 128)


def test_when_gdn2_backward_then_gradients_flow_to_parameters() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _, _ = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 8, 64)
    for name, param in layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_when_torch_backends_then_outputs_are_equivalent() -> None:
    """All pure-torch / compiled GDN-2 backends must agree numerically."""
    torch.manual_seed(0)
    made: dict[str, GatedDeltaNet2] = {}
    for be in ("torch", "torch-chunk", "torch-recurrent"):
        made[be] = GatedDeltaNet2(
            hidden_size=64,
            num_heads=2,
            head_dim=32,
            backend=be,
            chunk_size=8,
            use_short_conv=False,
        )
        made[be].eval()
    # Copy weights from a reference so the only difference is the backend.
    ref = made["torch"]
    for be, layer in made.items():
        if be != "torch":
            layer.load_state_dict(ref.state_dict())

    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        out_torch = made["torch"](x)[0]
        for be in ("torch-chunk", "torch-recurrent"):
            out_be = made[be](x)[0]
            assert torch.allclose(out_torch, out_be, atol=1e-5), be


def test_when_invalid_backend_then_raises() -> None:
    with pytest.raises(ValueError):
        GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, backend="nope")


def test_when_fp32_decay_then_forward_is_finite_and_shape_preserved() -> None:
    layer = GatedDeltaNet2(
        hidden_size=64, num_heads=2, head_dim=32, fp32_decay=True, chunk_size=8
    )
    x = torch.randn(2, 11, 64)
    out, _, _ = layer(x)
    assert out.shape == (2, 11, 64)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("chunk_size", [1, 8, 33])
@pytest.mark.parametrize("seq_len", [8, 41])
def test_when_chunk_and_recurrent_then_match_across_chunk_sizes(
    chunk_size: int, seq_len: int
) -> None:
    """The chunkwise (WY) implementation must match the sequential oracle."""
    b, t, h, d_k, d_v = 2, seq_len, 4, 16, 16
    torch.manual_seed(42)
    q = torch.randn(b, t, h, d_k)
    k = torch.randn(b, t, h, d_k)
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k).abs()
    erase_b = torch.sigmoid(torch.randn(b, t, h, d_k))
    write_w = torch.sigmoid(torch.randn(b, t, h, d_v))
    init = torch.randn(b, h, d_k, d_v) * 0.1

    out_rec, state_rec = torch_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=erase_b,
        w=write_w,
        initial_state=init,
        output_final_state=True,
    )
    out_chk, state_chk = torch_chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=erase_b,
        w=write_w,
        initial_state=init,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert out_chk.shape == (b, t, h, d_v)
    assert state_rec is not None and state_chk is not None
    assert torch.allclose(out_rec, out_chk, atol=1e-5)
    assert torch.allclose(state_rec, state_chk, atol=1e-5)


def test_when_chunk_and_recurrent_then_gradients_match() -> None:
    """Autograd through the chunkwise path must match the sequential oracle."""
    b, t, h, d_k, d_v = 1, 9, 2, 8, 8
    torch.manual_seed(7)
    tensors = (
        torch.randn(b, t, h, d_k, requires_grad=True),  # q
        torch.randn(b, t, h, d_k, requires_grad=True),  # k
        torch.randn(b, t, h, d_v, requires_grad=True),  # v
        -torch.rand(b, t, h, d_k).abs().requires_grad_(),  # g
        torch.sigmoid(torch.randn(b, t, h, d_k)).requires_grad_(),  # b
        torch.sigmoid(torch.randn(b, t, h, d_v)).requires_grad_(),  # w
    )

    def run(fn: Any, chunk_size: int | None) -> tuple[torch.Tensor, ...]:
        q, k, v, g, b, w = tensors
        kw: dict[str, Any] = {}
        if chunk_size is not None:
            kw["chunk_size"] = chunk_size
        out = fn(q, k, v, g, b, w, output_final_state=False, **kw)[0]
        return torch.autograd.grad(
            out.sum(), tensors, retain_graph=True, allow_unused=True
        )

    grads_rec = run(torch_recurrent_gdn2, None)
    grads_chk = run(torch_chunk_gdn2, 8)
    for name, gr, gc in zip(("q", "k", "v", "g", "b", "w"), grads_rec, grads_chk):
        assert gr is not None and gc is not None
        assert torch.allclose(gr, gc, atol=1e-5), f"gradient mismatch in {name}"


def test_when_chunk_incremental_then_matches_recurrent() -> None:
    """Chunking an already-running recurrent state must give identical outputs."""
    b, t, h, d_k, d_v = 2, 12, 3, 16, 16
    torch.manual_seed(3)
    q = torch.randn(b, t, h, d_k)
    k = torch.randn(b, t, h, d_k)
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k).abs()
    b_gate = torch.sigmoid(torch.randn(b, t, h, d_k))
    w_gate = torch.sigmoid(torch.randn(b, t, h, d_v))
    init = torch.randn(b, h, d_k, d_v) * 0.1

    split = 5
    q_pre, q_post = q[:, :split], q[:, split:]
    # Recurrent over the whole sequence, and recurrent-then-chunked.
    out_full, state_full = torch_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b_gate,
        w=w_gate,
        initial_state=init,
        output_final_state=True,
    )
    _, state_pre = torch_recurrent_gdn2(
        q=q_pre,
        k=k[:, :split],
        v=v[:, :split],
        g=g[:, :split],
        b=b_gate[:, :split],
        w=w_gate[:, :split],
        initial_state=init,
        output_final_state=True,
    )
    out_chk_post, state_post = torch_chunk_gdn2(
        q=q_post,
        k=k[:, split:],
        v=v[:, split:],
        g=g[:, split:],
        b=b_gate[:, split:],
        w=w_gate[:, split:],
        initial_state=state_pre,
        output_final_state=True,
        chunk_size=4,
    )
    assert torch.allclose(out_full[:, split:], out_chk_post, atol=1e-4)
    assert state_full is not None and state_post is not None
    assert torch.allclose(state_full, state_post, atol=1e-4)


def test_when_gdn2_forward_with_dict_cache_then_updates_state() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, layer_idx=0)
    x = torch.randn(1, 1, 64)
    cache: dict[
        str, torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}

    out, _, _ = layer(x, past_key_values=cache, use_cache=True)
    assert out.shape == (1, 1, 64)
    assert "recurrent_state" in cache
    assert "conv_state" in cache


def test_when_gdn2_forward_with_dynamic_cache_then_runs() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, layer_idx=0)
    x = torch.randn(1, 1, 64)
    dyn_cache = DynamicCache()

    out, _, past_cache = layer(x, past_key_values=dyn_cache, use_cache=True)
    assert out.shape == (1, 1, 64)
    assert past_cache is not None


def test_when_gdn2_registered_in_blocks_then_build_block_instantiates_gdn2() -> None:
    assert "gdn2" in BLOCKS
    assert "qwen35_gdn2" in BLOCKS
    cfg = DummyConfig(hidden_size=64, num_heads=2, head_dim=32)
    module = build_block("gdn2", cfg, layer_idx=0)
    assert isinstance(module, GatedDeltaNet2)


def test_when_gdn2_config_from_name_valid_then_returns_config() -> None:
    cfg = GDN2Config.from_name("gdn2_1.3B")
    assert cfg.hidden_size == 2304


def test_when_gdn2_host_initialized_then_parameters_have_correct_initial_values() -> (
    None
):
    from qwendopamine.models.gdn2.host import GDN2Host

    cfg = GDN2Config(hidden_size=64, num_heads=2, head_dim=32, num_layers=2)
    model = GDN2Host(cfg)

    # Embeddings initialized with std=0.02
    assert model.embed.weight.shape == (cfg.vocab_size, cfg.hidden_size)
    assert not torch.all(model.embed.weight == 0.0)

    # RMSNorm initialized with 1.0
    assert torch.allclose(model.norm.weight, torch.ones_like(model.norm.weight))

    # Check GDN2 layer initialization flags
    block0_attn = cast(GatedDeltaNet2, model.layers[0].attn)
    assert getattr(block0_attn.q_proj, "_is_hf_initialized", False)
    assert getattr(block0_attn.k_proj, "_is_hf_initialized", False)
    assert getattr(block0_attn.v_proj, "_is_hf_initialized", False)
    assert getattr(block0_attn.A_log, "_no_weight_decay", False)
    assert getattr(block0_attn.dt_bias, "_no_weight_decay", False)


def test_when_gdn2_forward_with_linear_attention_cache_layer_then_stores_states() -> (
    None
):
    try:
        from transformers.cache_utils import Cache, LinearAttentionCacheLayerMixin
    except ImportError:
        pytest.skip("LinearAttentionCacheLayerMixin not available")

    class TestLinearLayerCache(LinearAttentionCacheLayerMixin):
        """Tests for linear attention cache layer behavior."""

        def __init__(self) -> None:
            super().__init__(number_of_states=1)

        def lazy_initialization(
            self,
            conv_states: torch.Tensor | None = None,
            recurrent_states: torch.Tensor | None = None,
            state_idx: int = 0,
        ) -> None:
            pass

        def update_recurrent_state(
            self, recurrent_states: torch.Tensor, state_idx: int = 0, **kwargs: Any
        ) -> torch.Tensor:
            self.recurrent_states[state_idx] = recurrent_states
            return recurrent_states

        def update_conv_state(
            self, conv_states: Any, state_idx: int = 0, **kwargs: Any
        ) -> Any:
            if isinstance(conv_states, tuple):
                for i, st in enumerate(conv_states):
                    self.conv_states[i] = st
            else:
                self.conv_states[state_idx] = conv_states
            return conv_states

    layer_cache = TestLinearLayerCache()
    cache = Cache(layers=[layer_cache])

    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, layer_idx=0)
    x = torch.randn(1, 4, 64)

    out, _, past_cache = layer(x, past_key_values=cache, use_cache=True)
    assert out.shape == (1, 4, 64)
    assert past_cache is not None
    assert layer_cache.recurrent_states.get(0) is not None
    assert layer_cache.conv_states.get(0) is not None


def test_when_gdn2_incremental_decoding_then_updates_recurrent_state() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, layer_idx=0)
    x1 = torch.randn(1, 1, 64)
    x2 = torch.randn(1, 1, 64)

    cache: dict[
        str, torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}

    out1, _, _ = layer(x1, past_key_values=cache, use_cache=True)
    rec_state_1 = cache.get("recurrent_state")
    assert isinstance(rec_state_1, torch.Tensor)

    out2, _, _ = layer(x2, past_key_values=cache, use_cache=True)
    rec_state_2 = cache.get("recurrent_state")
    assert isinstance(rec_state_2, torch.Tensor)

    assert out1.shape == (1, 1, 64)
    assert out2.shape == (1, 1, 64)
    assert not torch.allclose(rec_state_1, rec_state_2)


def test_when_padded_attention_mask_then_padding_tokens_are_zeroed() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, chunk_size=8)
    x = torch.randn(2, 6, 64)
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.long
    )

    out, _, _ = layer(x, attention_mask=attention_mask)
    assert out.shape == (2, 6, 64)
    assert torch.all(out[1, 3:6] == 0.0)

    unmasked, _, _ = layer(x)
    assert not torch.allclose(out, unmasked)


def test_when_a_log_initialized_then_decay_rate_in_reference_range() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32)
    exp_a = layer.A_log.exp()
    assert bool((exp_a >= 1.0).all())
    assert bool((exp_a <= 16.0).all())


def test_when_fp32_decay_defaults_true_then_matches_reference() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32)
    assert layer.fp32_decay is True


def test_when_fp32_decay_default_and_bf16_then_forward_is_finite() -> None:
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32).to(torch.bfloat16)
    x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
    out, _, _ = layer(x)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_when_num_v_heads_not_divisible_then_raises() -> None:
    with pytest.raises(ValueError):
        GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, num_v_heads=3)
