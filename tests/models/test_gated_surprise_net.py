"""Behavioral and shape tests for hardware-agnostic GatedSurpriseNetAdam."""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from qwendopamine.blocks import GatedSurpriseNetAdam, GatedSurpriseNetBlock
from qwendopamine.integrations import (
    GatedSurpriseNetHFBlock,
    GatedSurpriseNetHFConfig,
    HFIntegration,
)
from qwendopamine.models.blocks.registry import BLOCKS, build_block
from qwendopamine.models.gated_surprise_net import (
    SurpriseMemory,
    SurpriseMemoryAdam,
    SurpriseRecurrenceState,
    gaussian_nll_diag,
    l2_normalize_last,
)
from qwendopamine.models.model_factory import ResearchDecoder


@pytest.fixture
def mock_surprise_config() -> types.SimpleNamespace:
    r"""Fixture providing configuration for GatedSurpriseNet testing."""
    return types.SimpleNamespace(
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        num_v_heads=2,
        expand_v=1.0,
        use_short_conv=True,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-5,
        local_adam_lr=1e-3,
        train_chunk_size=16,
        vocab_size=200,
        max_position_embeddings=128,
        num_layers=2,
        block_types=["gated_surprise_net", "surprise_net"],
        rms_norm_eps=1e-6,
    )


def test_when_gated_surprise_net_forward_cpu_then_preserves_shape(
    mock_surprise_config: types.SimpleNamespace,
) -> None:
    layer = GatedSurpriseNetAdam(
        hidden_size=mock_surprise_config.hidden_size,
        num_heads=mock_surprise_config.num_heads,
        head_dim=mock_surprise_config.head_dim,
    )
    x = torch.randn(2, 16, 64, dtype=torch.float32)
    out, attn, cache = layer(x)
    assert out.shape == (2, 16, 64)
    assert attn is None
    assert cache is None
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_when_gated_surprise_net_forward_dtypes_then_preserves_dtype_and_shape(
    dtype: torch.dtype,
) -> None:
    layer = GatedSurpriseNetAdam(hidden_size=64, num_heads=2, head_dim=32).to(dtype)
    x = torch.randn(2, 16, 64, dtype=dtype)
    out, attn, cache = layer(x)
    assert out.shape == (2, 16, 64)
    assert out.dtype == dtype
    assert attn is None
    assert cache is None
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("use_short_conv", [True, False])
@pytest.mark.parametrize("allow_neg_eigval", [True, False])
def test_when_gated_surprise_net_forward_variants_then_preserves_shape(
    use_short_conv: bool, allow_neg_eigval: bool
) -> None:
    layer = GatedSurpriseNetAdam(
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        use_short_conv=use_short_conv,
        allow_neg_eigval=allow_neg_eigval,
    )
    x = torch.randn(1, 8, 64)
    out, _, _ = layer(x)
    assert out.shape == (1, 8, 64)
    assert not torch.isnan(out).any()


def test_when_gated_surprise_net_backward_then_gradients_flow() -> None:
    layer = GatedSurpriseNetAdam(hidden_size=64, num_heads=2, head_dim=32)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _, _ = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 8, 64)
    for name, param in layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_when_gated_surprise_net_initialized_then_has_var_proj_with_zero_bias_for_unit_precision() -> (
    None
):
    layer = GatedSurpriseNetAdam(hidden_size=64, num_heads=2, head_dim=32)
    assert hasattr(layer, "var_proj")
    final_linear = layer.var_proj[-1]
    assert isinstance(final_linear, torch.nn.Linear)
    assert final_linear.bias is not None
    assert torch.allclose(final_linear.bias, torch.zeros_like(final_linear.bias))


def test_when_surprise_memory_adam_scans_then_serial_and_chunk_match() -> None:
    memory = SurpriseMemoryAdam(num_heads=2, head_k_dim=16, head_v_dim=16)
    b, t, h, d_k, d_v = 2, 16, 2, 16, 16

    q = l2_normalize_last(torch.randn(b, t, h, d_k))
    k = l2_normalize_last(torch.randn(b, t, h, d_k))
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k)
    erase_b = torch.sigmoid(torch.randn(b, t, h, d_k))
    write_w = torch.sigmoid(torch.randn(b, t, h, d_v))
    surprise_u = torch.sigmoid(torch.randn(b, t, h, d_v))

    out_serial, state_serial, nll_serial = memory.serial_scan(
        q, k, v, g, erase_b, write_w, surprise_u
    )
    out_chunk, state_chunk, nll_chunk = memory.chunk_parallel_training_scan(
        q, k, v, g, erase_b, write_w, surprise_u, chunk_size=8
    )

    assert out_serial.shape == (b, t, h, d_v)
    assert out_chunk.shape == (b, t, h, d_v)
    assert torch.allclose(out_serial, out_chunk, atol=1e-5)
    assert torch.allclose(state_serial.memory, state_chunk.memory, atol=1e-5)
    assert torch.allclose(nll_serial, nll_chunk, atol=1e-5)


def test_when_surprise_memory_non_multiple_chunk_size_then_matches_serial() -> None:
    memory = SurpriseMemory(num_heads=2, head_k_dim=16, head_v_dim=16)
    b, t, h, d_k, d_v = 2, 37, 2, 16, 16

    q = l2_normalize_last(torch.randn(b, t, h, d_k))
    k = l2_normalize_last(torch.randn(b, t, h, d_k))
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k)
    erase_b = torch.sigmoid(torch.randn(b, t, h, d_k))
    write_w = torch.sigmoid(torch.randn(b, t, h, d_v))
    surprise_u = torch.sigmoid(torch.randn(b, t, h, d_v))

    out_serial, state_serial, nll_serial = memory.serial_scan(
        q, k, v, g, erase_b, write_w, surprise_u
    )
    out_chunk, state_chunk, nll_chunk = memory.chunk_parallel_training_scan(
        q, k, v, g, erase_b, write_w, surprise_u, chunk_size=16
    )

    assert out_serial.shape == (b, t, h, d_v)
    assert out_chunk.shape == (b, t, h, d_v)
    assert torch.allclose(out_serial, out_chunk, atol=1e-5)
    assert torch.allclose(state_serial.memory, state_chunk.memory, atol=1e-5)
    assert torch.allclose(nll_serial, nll_chunk, atol=1e-5)


def test_when_surprise_gate_scaled_then_modulates_memory_state() -> None:
    memory = SurpriseMemory(num_heads=1, head_k_dim=8, head_v_dim=8)
    b, t, h, d_k, d_v = 1, 4, 1, 8, 8

    q = l2_normalize_last(torch.randn(b, t, h, d_k))
    k = l2_normalize_last(torch.randn(b, t, h, d_k))
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k) * 0.1
    b_gate = torch.ones(b, t, h, d_k) * 0.5
    w_gate = torch.ones(b, t, h, d_v) * 0.5

    u_zero = torch.zeros(b, t, h, d_v)
    u_one = torch.ones(b, t, h, d_v)

    out_z, state_z, _ = memory.serial_scan(q, k, v, g, b_gate, w_gate, u_zero)
    out_1, state_1, _ = memory.serial_scan(q, k, v, g, b_gate, w_gate, u_one)

    assert not torch.allclose(state_z.memory, state_1.memory)
    assert not torch.allclose(out_z, out_1)


def test_when_gated_surprise_net_forward_with_dict_cache_then_updates_states() -> None:
    layer = GatedSurpriseNetAdam(hidden_size=64, num_heads=2, head_dim=32, layer_idx=0)
    x = torch.randn(1, 1, 64)
    cache: dict[str, Any] = {}

    out, _, _ = layer(x, past_key_values=cache, use_cache=True)
    assert out.shape == (1, 1, 64)
    assert "recurrent_state" in cache
    assert isinstance(cache["recurrent_state"], SurpriseRecurrenceState)
    assert "conv_state" in cache


def test_when_gated_surprise_net_forward_with_dynamic_cache_then_runs() -> None:
    layer = GatedSurpriseNetAdam(hidden_size=64, num_heads=2, head_dim=32, layer_idx=0)
    x = torch.randn(1, 1, 64)
    dyn_cache = DynamicCache()

    out, _, past_cache = layer(x, past_key_values=dyn_cache, use_cache=True)
    assert out.shape == (1, 1, 64)
    assert past_cache is not None


def test_when_gated_surprise_net_autoregressive_step_then_matches_prefill() -> None:
    layer = GatedSurpriseNetAdam(hidden_size=64, num_heads=2, head_dim=32, layer_idx=0)
    layer.eval()

    x_full = torch.randn(1, 3, 64)
    out_prefill, _, _ = layer(x_full)

    cache: dict[str, Any] = {}
    out_steps = []
    for t in range(3):
        x_t = x_full[:, t : t + 1]
        out_t, _, _ = layer(x_t, past_key_values=cache, use_cache=True)
        out_steps.append(out_t)

    out_step_cat = torch.cat(out_steps, dim=1)
    assert out_step_cat.shape == (1, 3, 64)
    assert torch.allclose(out_prefill, out_step_cat, atol=1e-5)


def test_when_registered_in_blocks_then_build_block_instantiates_gated_surprise_net(
    mock_surprise_config: types.SimpleNamespace,
) -> None:
    assert "gated_surprise_net" in BLOCKS
    assert "surprise_net" in BLOCKS
    assert "qwen35_gated_surprise_net" in BLOCKS

    module1 = build_block("gated_surprise_net", mock_surprise_config, layer_idx=0)
    module2 = build_block("surprise_net", mock_surprise_config, layer_idx=1)

    assert isinstance(module1, GatedSurpriseNetAdam)
    assert isinstance(module2, GatedSurpriseNetAdam)
    assert isinstance(module1, GatedSurpriseNetBlock)


def test_when_research_decoder_uses_gated_surprise_net_then_executes(
    mock_surprise_config: types.SimpleNamespace,
) -> None:
    model = ResearchDecoder(mock_surprise_config)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits = model(input_ids)
    assert logits.shape == (2, 3, mock_surprise_config.vocab_size)
    assert not torch.isnan(logits).any()


def test_when_hf_config_and_block_instantiated_then_forward_succeeds() -> None:
    HFIntegration.register_gdn2_hf()
    hf_config = GatedSurpriseNetHFConfig(hidden_size=64, num_heads=2, head_dim=32)
    assert hf_config.model_type == "gated_surprise_net"

    hf_block = GatedSurpriseNetHFBlock(hf_config, layer_idx=0)
    x = torch.randn(1, 4, 64)
    out, _, _ = hf_block(x)
    assert out.shape == (1, 4, 64)


def test_when_gaussian_nll_diag_called_then_returns_correct_shape() -> None:
    target = torch.randn(2, 4, 16)
    mean = torch.randn(2, 4, 16)
    var = torch.rand(2, 4, 16).clamp_min(1e-3)

    loss = gaussian_nll_diag(target, mean, var)
    assert loss.shape == (2, 4)
    assert (loss >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_when_gated_surprise_net_cuda_then_runs_on_device() -> None:
    device = torch.device("cuda:0")
    layer = GatedSurpriseNetAdam(hidden_size=64, num_heads=2, head_dim=32).to(device)
    x = torch.randn(2, 8, 64, device=device)

    out, _, _ = layer(x)
    assert out.device == device
    assert out.shape == (2, 8, 64)


def test_when_hybrid_gpt_config_presets_resolved_then_matches_1b_spec() -> None:
    from qwendopamine.models.surprise_gpt import (
        SurpriseGPTConfig,
        compute_model_params,
    )

    cfg_1b = SurpriseGPTConfig.from_name("1B")
    assert cfg_1b.n_layer == 24
    assert cfg_1b.n_embd == 2048
    assert cfg_1b.n_head == 16
    assert cfg_1b.n_query_groups == 8
    assert cfg_1b.head_size == 128
    assert cfg_1b.intermediate_size == 5504
    assert cfg_1b.padded_vocab_size == cfg_1b.vocab_size

    stats = compute_model_params(cfg_1b)
    assert 1.3e9 <= stats["total"] <= 1.4e9
    assert stats["num_standard_layers"] == 23
    assert stats["num_surprise_layers"] == 1


def test_when_hybrid_gpt_config_invalid_preset_then_raises_key_error() -> None:
    from qwendopamine.models.surprise_gpt import SurpriseGPTConfig

    with pytest.raises(KeyError, match="Unknown config name"):
        SurpriseGPTConfig.from_name("invalid_preset_name")


def test_when_causal_self_attention_incompatible_heads_then_raises_value_error() -> (
    None
):
    from qwendopamine.models.surprise_gpt import (
        CausalSelfAttention,
        SurpriseGPTConfig,
    )

    bad_cfg = SurpriseGPTConfig(n_embd=256, n_head=7, n_query_groups=4, head_size=32)
    with pytest.raises(ValueError, match="divisible by n_query_groups"):
        CausalSelfAttention(bad_cfg, layer_idx=0, n_embd=256)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_when_apply_rotary_emb_called_then_preserves_exact_dtype(
    dtype: torch.dtype,
) -> None:
    from qwendopamine.models.surprise_gpt import (
        apply_rotary_emb,
        build_rope_cache,
    )

    device = torch.device("cpu")
    cos, sin = build_rope_cache(16, 32, torch.float32, device)
    q = torch.randn(2, 16, 4, 32, dtype=dtype)
    q_rot = apply_rotary_emb(q, cos, sin)

    assert q_rot.dtype == dtype
    assert q_rot.shape == q.shape
    assert not torch.isnan(q_rot.float()).any()


def test_when_block_constructed_then_routes_center_layer_to_surprise_net() -> None:
    from qwendopamine.models.surprise_gpt import Block, SurpriseGPTConfig

    cfg4 = SurpriseGPTConfig.from_name("tiny", n_layer=4)
    assert Block(cfg4, 0).use_surprise_net is False
    assert Block(cfg4, 1).use_surprise_net is False
    assert Block(cfg4, 2).use_surprise_net is True
    assert Block(cfg4, 3).use_surprise_net is False

    cfg6 = SurpriseGPTConfig.from_name("small", n_layer=6)
    assert Block(cfg6, 3).use_surprise_net is True
    assert Block(cfg6, 0).use_surprise_net is False


def test_when_hybrid_gpt_forward_backward_then_loss_and_gradients_compute() -> None:
    from qwendopamine.models.surprise_gpt import SurpriseGPT, SurpriseGPTConfig

    cfg = SurpriseGPTConfig.from_name(
        "tiny", vocab_size=100, block_size=32, train_chunk_size=16
    )
    model = SurpriseGPT(cfg)
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))

    logits = model(x)
    assert logits.shape == (2, 16, 100)
    assert not torch.isnan(logits).any()

    loss = F.cross_entropy(logits.reshape(-1, 100), y.reshape(-1))
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} missing gradient"
            assert not torch.isnan(param.grad).any(), (
                f"Parameter {name} has NaN gradient"
            )


def test_when_chunk_gated_surprise_net_op_called_then_matches_reference() -> None:
    from qwendopamine.models.gated_surprise_net_ops import chunk_gated_surprise_net

    bs, ts, num_heads, head_k_dim, head_v_dim = 2, 16, 2, 16, 16
    q = torch.randn(bs, ts, num_heads, head_k_dim)
    k = torch.randn(bs, ts, num_heads, head_k_dim)
    v = torch.randn(bs, ts, num_heads, head_v_dim)
    g = -torch.rand(bs, ts, num_heads, head_k_dim)
    b = torch.sigmoid(torch.randn(bs, ts, num_heads, head_k_dim))
    w = torch.sigmoid(torch.randn(bs, ts, num_heads, head_v_dim))
    pi = 2.0 * torch.sigmoid(torch.randn(bs, ts, num_heads, head_v_dim))

    out, final_state = chunk_gated_surprise_net(
        q=q, k=k, v=v, g=g, b=b, w=w, pi=pi, chunk_size=8
    )

    mem = SurpriseMemory(
        hidden_size=num_heads * head_k_dim,
        num_heads=num_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )
    sigma_sq = 1.0 / pi.clamp_min(1e-6)
    out_ref, final_state_ref, _ = mem.chunk_parallel_training_scan(
        q=q, k=k, v=v, g=g, b=b, w=w, sigma_sq=sigma_sq, chunk_size=8
    )

    assert out.shape == (bs, ts, num_heads, head_v_dim)
    assert final_state.shape == (bs, num_heads, head_k_dim, head_v_dim)
    assert not torch.isnan(out).any()
    assert not torch.isnan(final_state).any()

    assert torch.allclose(out, out_ref, atol=1e-5)
    assert torch.allclose(final_state, final_state_ref.memory, atol=1e-5)


def test_when_gated_surprise_net_forward_at_full_train_chunk_size_then_output_is_finite() -> (
    None
):
    # Behavioral contract: the per-chunk cumulative decay gamma must not
    # underflow to zero in float32 during a training forward pass.  When
    # A_log ~ uniform_(1, 16), A reaches up to 16, and over 128 tokens
    # cumsum(g) ≈ -A * 128 * E[softplus] ≈ -6000, so exp(cumsum) = 0 in
    # float32.  kbar = (b*k) / gamma then saturates at 1/clamp_min = 1e12,
    # silently discarding all memory state information for tokens deep in the
    # chunk.  The fix is A_log ~ uniform_(0.1, 0.5), keeping cumsum(g) in a
    # range where gamma stays well above float32 underflow (~1e-38).
    import torch.nn.functional as F

    train_chunk_size = 128
    layer = GatedSurpriseNetAdam(
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        train_chunk_size=train_chunk_size,
    )
    layer.eval()
    torch.manual_seed(0)
    x = torch.randn(1, train_chunk_size, 64)
    with torch.no_grad():
        g = (
            -layer.A_log.float().exp().repeat_interleave(layer.head_k_dim)
            * F.softplus(layer.f_proj(x).float() + layer.dt_bias)
        ).view(1, train_chunk_size, layer.num_heads, layer.head_k_dim)
        gamma = torch.exp(torch.cumsum(g, dim=1))
    # gamma must stay above float32 underflow floor so kbar = k/gamma
    # does not amplify noise to 1e12, silently wiping the memory state.
    assert gamma.min().item() > 1e-35, (
        f"Cumulative decay gamma underflowed to {gamma.min().item():.2e} within "
        f"a {train_chunk_size}-token chunk.  This silently corrupts chunk-scan "
        "memory state.  Fix: reduce A_log initialization range."
    )
