"""Behavioral and shape tests for hardware-agnostic GatedSurpriseNetAdam."""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch
from transformers.cache_utils import DynamicCache

from qwendopamine.blocks import GatedSurpriseNetAdam, GatedSurpriseNetBlock
from qwendopamine.integrations import (
    GatedSurpriseNetHFBlock,
    GatedSurpriseNetHFConfig,
    HFIntegration,
)
from qwendopamine.models.blocks.registry import BLOCKS, build_block
from qwendopamine.models.gated_surprise_net import (
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


def test_when_surprise_memory_adam_scans_then_serial_and_chunk_match() -> None:
    memory = SurpriseMemoryAdam(num_heads=2, head_k_dim=16, head_v_dim=16)
    b, t, h, d_k, d_v = 2, 16, 2, 16, 16

    q = l2_normalize_last(torch.randn(b, t, h, d_k))
    k = l2_normalize_last(torch.randn(b, t, h, d_k))
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k)
    erase_b = torch.sigmoid(torch.randn(b, t, h, d_k))
    write_w = torch.sigmoid(torch.randn(b, t, h, d_v))

    out_serial, state_serial, nll_serial = memory.serial_scan(q, k, v, g, erase_b, write_w)
    out_chunk, state_chunk, nll_chunk = memory.chunk_parallel_training_scan(
        q, k, v, g, erase_b, write_w, chunk_size=8
    )

    assert out_serial.shape == (b, t, h, d_v)
    assert out_chunk.shape == (b, t, h, d_v)
    assert torch.allclose(out_serial, out_chunk, atol=1e-5)
    assert torch.allclose(state_serial.memory, state_chunk.memory, atol=1e-5)
    assert torch.allclose(nll_serial, nll_chunk, atol=1e-5)


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
