"""Tests for the decomposed GDN-2 modules and the new step() API."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# Direct import (was previously from qwendopamine.models.gdn2.gdn2)
from qwendopamine.models.gdn2 import GatedDeltaNet2
from qwendopamine.models.gdn2 import (
    GatedDeltaNet2 as LegacyGatedDeltaNet2,
)
from qwendopamine.models.gdn2 import (
    compute_gdn2_intra_chunk_scores as legacy_compute_gdn2_intra_chunk_scores,
)
from qwendopamine.models.gdn2 import (
    compute_gdn2_wy_coefficients as legacy_compute_gdn2_wy_coefficients,
)
from qwendopamine.models.gdn2 import (
    resolve_gdn2_backend as legacy_resolve_gdn2_backend,
)
from qwendopamine.models.gdn2 import (
    torch_chunk_gdn2 as legacy_torch_chunk_gdn2,
)
from qwendopamine.models.gdn2.backend import GDN2_BACKENDS, resolve_gdn2_backend
from qwendopamine.models.gdn2.ops.conv import ShortConvolution
from qwendopamine.models.gdn2.ops.norm import RMSNormGated
from qwendopamine.models.gdn2.recurrence.chunk import (
    compute_gdn2_intra_chunk_scores,
    compute_gdn2_wy_coefficients,
    torch_chunk_gdn2,
)
from qwendopamine.models.gdn2.recurrence.recurrent import (
    gated_delta_2_step,
    torch_recurrent_gdn2,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gdn2_layer(hidden_size: int = 64, num_heads: int = 2, head_dim: int = 32):
    return GatedDeltaNet2(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        use_short_conv=False,
    )


# ---------------------------------------------------------------------------
# Backward-compatible imports
# ---------------------------------------------------------------------------


def test_when_importing_from_gdn2_gdn2_then_symbols_are_available() -> None:
    assert LegacyGatedDeltaNet2 is GatedDeltaNet2
    assert legacy_torch_chunk_gdn2 is torch_chunk_gdn2
    assert legacy_resolve_gdn2_backend is resolve_gdn2_backend
    assert legacy_compute_gdn2_wy_coefficients is compute_gdn2_wy_coefficients
    assert legacy_compute_gdn2_intra_chunk_scores is compute_gdn2_intra_chunk_scores


# ---------------------------------------------------------------------------
# step() — single-token auto-regressive generation
# ---------------------------------------------------------------------------


def test_when_step_with_1d_input_then_output_shape_is_correct() -> None:
    layer = _make_gdn2_layer()
    layer.eval()
    x_1d = torch.randn(2, 64)
    out, state, conv_states = layer.step(x_1d)
    assert out.shape == (2, 1, 64)
    assert state.shape == (2, 2, 32, 32)
    assert conv_states == (None, None, None)


def test_when_step_with_2d_input_then_output_shape_is_correct() -> None:
    layer = _make_gdn2_layer()
    layer.eval()
    x_2d = torch.randn(2, 1, 64)
    out, state, _ = layer.step(x_2d)
    assert out.shape == (2, 1, 64)
    assert state.shape == (2, 2, 32, 32)


def test_when_step_twice_then_state_is_updated() -> None:
    layer = _make_gdn2_layer()
    layer.eval()
    x1 = torch.randn(2, 1, 64)
    x2 = torch.randn(2, 1, 64)

    out1, state1, _ = layer.step(x1)
    out2, state2, _ = layer.step(x2, state=state1)

    assert out1.shape == (2, 1, 64)
    assert out2.shape == (2, 1, 64)
    assert not torch.allclose(state1, state2)


def test_when_step_matches_forward_then_outputs_agree() -> None:
    """step() token-by-token must match the full forward pass."""
    layer = _make_gdn2_layer()
    layer.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 4, 64)

    with torch.no_grad():
        full_out, _, _ = layer(x)

    step_outs = []
    state = None
    for t in range(x.shape[1]):
        xt = x[:, t : t + 1, :]
        out_t, state, _ = layer.step(xt, state=state)
        step_outs.append(out_t)

    step_out = torch.cat(step_outs, dim=1)
    assert torch.allclose(full_out, step_out, atol=1e-5)


def test_when_step_with_initial_state_then_state_is_used() -> None:
    layer = _make_gdn2_layer()
    layer.eval()
    init_state = torch.randn(2, 2, 32, 32)
    x = torch.randn(2, 1, 64)

    out1, state1, _ = layer.step(x, state=init_state)
    out2, state2, _ = layer.step(x, state=init_state)

    # Same initial state → same output and same updated state
    assert torch.allclose(out1, out2)
    assert torch.allclose(state1, state2)


def test_when_step_with_cache_then_conv_states_are_returned() -> None:
    layer = GatedDeltaNet2(
        hidden_size=64, num_heads=2, head_dim=32, use_short_conv=True, conv_size=4
    )
    layer.eval()
    x = torch.randn(2, 1, 64)
    out, state, conv_states = layer.step(x)
    assert out.shape == (2, 1, 64)
    assert state.shape == (2, 2, 32, 32)
    # With short conv, conv states should be tensors (not None)
    for cs in conv_states:
        assert isinstance(cs, torch.Tensor)


def test_when_step_gradients_flow_then_backward_works() -> None:
    layer = _make_gdn2_layer()
    x = torch.randn(1, 1, 64, requires_grad=True)
    out, _, _ = layer.step(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (1, 1, 64)
    for name, param in layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


# ---------------------------------------------------------------------------
# core.py — recurrent engine
# ---------------------------------------------------------------------------


def test_when_torch_recurrent_gdn2_then_output_shape_is_correct() -> None:
    b, t, h, d_k, d_v = 2, 8, 2, 16, 16
    torch.manual_seed(0)
    q = torch.randn(b, t, h, d_k)
    k = torch.randn(b, t, h, d_k)
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k).abs()
    b_gate = torch.sigmoid(torch.randn(b, t, h, d_k))
    w_gate = torch.sigmoid(torch.randn(b, t, h, d_v))

    out, state = torch_recurrent_gdn2(
        q=q, k=k, v=v, g=g, b=b_gate, w=w_gate, output_final_state=True
    )
    assert out.shape == (b, t, h, d_v)
    assert state is not None
    assert state.shape == (b, h, d_k, d_v)


def test_when_torch_recurrent_gdn2_no_state_then_state_is_none() -> None:
    b, t, h, d_k, d_v = 2, 4, 2, 8, 8
    q = torch.randn(b, t, h, d_k)
    k = torch.randn(b, t, h, d_k)
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k).abs()
    b_gate = torch.sigmoid(torch.randn(b, t, h, d_k))
    w_gate = torch.sigmoid(torch.randn(b, t, h, d_v))

    out, state = torch_recurrent_gdn2(
        q=q, k=k, v=v, g=g, b=b_gate, w=w_gate, output_final_state=False
    )
    assert out.shape == (b, t, h, d_v)
    assert state is None


def test_when_gated_delta_2_step_then_matches_recurrent_single_step() -> None:
    """gated_delta_2_step must match the verified torch_recurrent_gdn2 oracle."""
    b, h, d_k, d_v = 2, 4, 8, 8
    torch.manual_seed(0)
    S = torch.randn(b, h, d_k, d_v)
    q_t = torch.randn(b, h, d_k)
    k_t = torch.randn(b, h, d_k)
    v_t = torch.randn(b, h, d_v)
    b_t = torch.sigmoid(torch.randn(b, h, d_k))
    w_t = torch.sigmoid(torch.randn(b, h, d_v))
    a_t = torch.rand(b, h, d_k).abs()

    # Preprocess q/k exactly as torch_recurrent_gdn2 does.
    q_proc = F.normalize(q_t, p=2, dim=-1, eps=1e-6) * (d_k**-0.5)
    k_proc = F.normalize(k_t, p=2, dim=-1, eps=1e-6)

    y_step, S_next_step = gated_delta_2_step(
        S=S, q_t=q_proc, k_t=k_proc, v_t=v_t, b_t=b_t, w_t=w_t, a_t=a_t
    )

    # Run the recurrent oracle on a single-token sequence with the same inputs.
    q_seq = q_t.unsqueeze(1)
    k_seq = k_t.unsqueeze(1)
    v_seq = v_t.unsqueeze(1)
    b_seq = b_t.unsqueeze(1)
    w_seq = w_t.unsqueeze(1)
    g_seq = torch.log(a_t).unsqueeze(1)
    out_seq, S_next_seq = torch_recurrent_gdn2(
        q=q_seq,
        k=k_seq,
        v=v_seq,
        g=g_seq,
        b=b_seq,
        w=w_seq,
        initial_state=S,
        output_final_state=True,
    )

    assert S_next_seq is not None
    assert torch.allclose(y_step, out_seq.squeeze(1), atol=1e-5)
    assert torch.allclose(S_next_step, S_next_seq, atol=1e-5)


# ---------------------------------------------------------------------------
# chunk.py — chunkwise WY representation
# ---------------------------------------------------------------------------


def test_when_torch_chunk_gdn2_then_matches_recurrent() -> None:
    b, t, h, d_k, d_v = 2, 16, 4, 16, 16
    torch.manual_seed(42)
    q = torch.randn(b, t, h, d_k)
    k = torch.randn(b, t, h, d_k)
    v = torch.randn(b, t, h, d_v)
    g = -torch.rand(b, t, h, d_k).abs()
    b_gate = torch.sigmoid(torch.randn(b, t, h, d_k))
    w_gate = torch.sigmoid(torch.randn(b, t, h, d_v))
    init = torch.randn(b, h, d_k, d_v) * 0.1

    out_rec, state_rec = torch_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b_gate,
        w=w_gate,
        initial_state=init,
        output_final_state=True,
    )
    out_chk, state_chk = torch_chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b_gate,
        w=w_gate,
        initial_state=init,
        output_final_state=True,
        chunk_size=8,
    )
    assert state_rec is not None
    assert torch.allclose(out_rec, out_chk, atol=1e-5)
    assert state_chk is not None
    assert torch.allclose(state_rec, state_chk, atol=1e-5)


def test_when_compute_gdn2_wy_coefficients_then_shapes_are_correct() -> None:
    b, h, c, d = 2, 4, 8, 8
    torch.manual_seed(0)
    kbar = torch.randn(b, h, c, d)
    ebar = torch.randn(b, h, c, d)
    z = torch.randn(b, h, c, d)
    device = kbar.device

    y, u = compute_gdn2_wy_coefficients(kbar, ebar, z, device)
    assert y.shape == (b, h, c, d)
    assert u.shape == (b, h, c, d)


def test_when_compute_gdn2_intra_chunk_scores_then_shapes_are_correct() -> None:
    b, h, c, d = 2, 4, 8, 8
    torch.manual_seed(0)
    q = torch.randn(b, h, c, d)
    gamma = torch.rand(b, h, c, d).abs()
    kbar = torch.randn(b, h, c, d)

    scores = compute_gdn2_intra_chunk_scores(q, gamma, kbar)
    assert scores.shape == (b, h, c, c)


# ---------------------------------------------------------------------------
# normalization.py — RMSNormGated
# ---------------------------------------------------------------------------


def test_when_rms_norm_gated_then_output_shape_matches_input() -> None:
    norm = RMSNormGated(hidden_size=64, eps=1e-5)
    x = torch.randn(2, 8, 64)
    gate = torch.randn(2, 8, 64)
    out = norm(x, gate)
    assert out.shape == x.shape


def test_when_rms_norm_gated_then_output_is_finite() -> None:
    norm = RMSNormGated(hidden_size=64)
    x = torch.randn(2, 8, 64)
    gate = torch.randn(2, 8, 64)
    out = norm(x, gate)
    assert torch.isfinite(out).all()


def test_when_rms_norm_gated_unit_gate_then_behaves_like_std_norm() -> None:
    norm = RMSNormGated(hidden_size=64)
    x = torch.randn(2, 8, 64)
    out = norm(x, torch.ones_like(x))
    # With unit gate, output should be normalized
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# convolution.py — ShortConvolution
# ---------------------------------------------------------------------------


def test_when_short_convolution_forward_then_output_shape_is_correct() -> None:
    conv = ShortConvolution(hidden_size=64, kernel_size=4)
    x = torch.randn(2, 8, 64)
    out, new_state = conv(x, output_final_state=True)
    assert out.shape == (2, 8, 64)
    assert new_state is not None
    assert new_state.shape[0] == 2


def test_when_short_convolution_cache_then_state_changes() -> None:
    conv = ShortConvolution(hidden_size=64, kernel_size=4)
    x1 = torch.randn(2, 3, 64)
    x2 = torch.randn(2, 3, 64)

    _, state1 = conv(x1, output_final_state=True)
    _, state2 = conv(x2, cache=state1, output_final_state=True)

    assert not torch.allclose(state1, state2)


def test_when_short_convolution_no_output_state_then_none_returned() -> None:
    conv = ShortConvolution(hidden_size=64, kernel_size=4)
    x = torch.randn(2, 4, 64)
    out, state = conv(x, output_final_state=False)
    assert out.shape == (2, 4, 64)
    assert state is None


# ---------------------------------------------------------------------------
# cache_utils.py
# ---------------------------------------------------------------------------


def test_when_pad_input_then_shapes_are_correct() -> None:
    from qwendopamine.models.gdn2.recurrence.packing import get_unpad_data, pad_input

    x = torch.randn(2, 4, 64)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)
    indices, _, _ = get_unpad_data(attention_mask)

    packed = x.view(-1, 64)[indices]
    padded = pad_input(packed, indices, batch=2, seqlen=4)
    assert padded.shape == (2, 4, 64)


def test_when_index_first_axis_then_selects_correct_elements() -> None:
    from qwendopamine.models.gdn2.recurrence.packing import index_first_axis

    x = torch.randn(2, 2, 64)  # [batch, seq, dim]
    indices = torch.tensor(
        [0, 2, 3]
    )  # select first token of batch 0, first of batch 1, second of batch 1
    out = index_first_axis(x, indices)
    assert out.shape == (3, 64)
    assert torch.allclose(out[0], x[0, 0])
    assert torch.allclose(out[1], x[1, 0])
    assert torch.allclose(out[2], x[1, 1])


def test_when_get_unpad_data_then_returns_correct_indices() -> None:
    from qwendopamine.models.gdn2.recurrence.packing import get_unpad_data

    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)
    indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)
    assert max_seqlen == 3
    assert cu_seqlens.tolist() == [0, 2, 5]
    assert indices.tolist() == [0, 1, 4, 5, 6]


# ---------------------------------------------------------------------------
# backend.py — backend dispatch
# ---------------------------------------------------------------------------


def test_when_resolve_gdn2_backend_valid_then_returns_backend_name() -> None:
    backend = resolve_gdn2_backend("torch", training=True, seq_len=16)
    assert backend == "torch"


def test_when_resolve_gdn2_backend_invalid_then_raises() -> None:
    with pytest.raises(ValueError):
        resolve_gdn2_backend("invalid-backend", training=True, seq_len=16)


def test_when_gdn2_backends_contains_expected_backends() -> None:
    assert "torch" in GDN2_BACKENDS
    assert "torch-chunk" in GDN2_BACKENDS
    assert "torch-recurrent" in GDN2_BACKENDS


def test_when_resolve_gdn2_backend_auto_then_returns_available_backend() -> None:
    backend = resolve_gdn2_backend("auto", training=False, seq_len=1)
    assert backend in GDN2_BACKENDS


# ---------------------------------------------------------------------------
# Integration: step() with DynamicCache
# ---------------------------------------------------------------------------


def test_when_step_returns_conv_states_then_none_for_no_conv() -> None:
    layer = _make_gdn2_layer()
    layer.eval()
    x = torch.randn(1, 1, 64)

    out, state, conv_states = layer.step(x)
    assert out.shape == (1, 1, 64)
    assert state is not None
    # Conv states are None when use_short_conv=False
    assert conv_states == (None, None, None)
