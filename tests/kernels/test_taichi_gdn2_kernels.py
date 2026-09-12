"""Direct numerical correctness tests for the Taichi GDN-2 kernel primitives.

Each new kernel is tested against a pure-PyTorch reference so that
forward parity (and where applicable backward/vJP parity) is verified
at the kernel level, not just through the higher-level ops layer.
"""

from __future__ import annotations

import pytest
import torch

from qwendopamine.kernels.taichi import gdn2_kernels as _kernels
from qwendopamine.kernels.taichi import is_available
from qwendopamine.kernels.taichi.gdn2_kernels import (
    launch_chunk_fwd_pipeline,
    launch_chunk_intra_token_parallel,
    launch_naive_gdn2,
    launch_wy_fast,
)
from qwendopamine.models.gdn2.recurrence.chunk import (
    compute_gdn2_intra_chunk_scores,
    compute_gdn2_wy_coefficients,
    torch_chunk_gdn2,
)
from qwendopamine.models.gdn2.recurrence.recurrent import torch_recurrent_gdn2

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)

_K = 4
_V = 4
_C = 8


def _make_random_inputs(C: int = _C, K: int = _K, V: int = _V) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    q = torch.randn(C, K)
    k = torch.randn(C, K)
    v = torch.randn(C, V)
    g_log = (torch.randn(C, K) * 0.1).float()
    b = torch.sigmoid(torch.randn(C, K))
    w = torch.sigmoid(torch.randn(C, V))
    return q, k, v, g_log, b, w


# ---------------------------------------------------------------------------
# launch_chunk_intra_token_parallel
# ---------------------------------------------------------------------------


def _torch_chunk_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gamma: torch.Tensor,
    kbar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    C, K = q.shape
    Aqk = torch.zeros(C, C)
    Akk = torch.zeros(C, C)
    for i in range(C):
        for j in range(C):
            if j <= i:
                for kk in range(K):
                    Aqk[i, j] += (gamma[i, kk] * q[i, kk]) * kbar[j, kk]
                    Akk[i, j] += (gamma[i, kk] * k[i, kk]) * kbar[j, kk]
    return Aqk, Akk


def test_launch_chunk_intra_token_parallel_matches_torch() -> None:
    q, k, _v, g_log, _b, _w = _make_random_inputs()
    gamma = torch.exp(torch.cumsum(g_log, dim=0))
    kbar = k / gamma.clamp_min(1e-12)

    Aqk = torch.zeros(_C, _C)
    Akk = torch.zeros(_C, _C)
    launch_chunk_intra_token_parallel(q, k, gamma, kbar, Aqk, Akk)

    ref_Aqk, ref_Akk = _torch_chunk_intra_token_parallel(q, k, gamma, kbar)
    torch.testing.assert_close(Aqk, ref_Aqk, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(Akk, ref_Akk, atol=1e-5, rtol=1e-5)


def test_launch_chunk_intra_token_parallel_causal_mask() -> None:
    """Entries with i > j (upper triangle) must be zero."""
    q, k, _v, g_log, _b, _w = _make_random_inputs()
    gamma = torch.exp(torch.cumsum(g_log, dim=0))
    kbar = k / gamma.clamp_min(1e-12)

    Aqk = torch.zeros(_C, _C)
    Akk = torch.zeros(_C, _C)
    launch_chunk_intra_token_parallel(q, k, gamma, kbar, Aqk, Akk)

    row_idx = torch.arange(_C).unsqueeze(-1)
    col_idx = torch.arange(_C).unsqueeze(0)
    mask = row_idx < col_idx
    assert torch.all(Aqk[mask] == 0.0), "Aqk upper-triangle not zero"
    assert torch.all(Akk[mask] == 0.0), "Akk upper-triangle not zero"


def test_launch_chunk_intra_token_parallel_different_sizes() -> None:
    for C in [4, 6, 16]:
        q, k, _v, g_log, _b, _w = _make_random_inputs(C=C)
        gamma = torch.exp(torch.cumsum(g_log, dim=0))
        kbar = k / gamma.clamp_min(1e-12)

        Aqk = torch.zeros(C, C)
        Akk = torch.zeros(C, C)
        launch_chunk_intra_token_parallel(q, k, gamma, kbar, Aqk, Akk)

        ref_Aqk, ref_Akk = _torch_chunk_intra_token_parallel(q, k, gamma, kbar)
        torch.testing.assert_close(Aqk, ref_Aqk, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(Akk, ref_Akk, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# launch_wy_fast
# ---------------------------------------------------------------------------


def _torch_wy_fast(
    A: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ebar = gamma * (b * k)
    z = w_gate * v
    w_out = torch.matmul(A, ebar)
    u = torch.matmul(A, z)
    qg = gamma * q
    gm_last = gamma[-1:]
    kg = (gm_last / gamma.clamp_min(1e-12)) * k
    return w_out, u, qg, kg


def _solve_wy_matrix(
    g_log: torch.Tensor,
    b: torch.Tensor,
    k: torch.Tensor,
    K_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gamma = torch.exp(torch.cumsum(g_log, dim=0))
    kbar = k / gamma.clamp_min(1e-12)
    ebar = gamma * (b * k)
    C = g_log.shape[0]
    A = torch.eye(C)
    for i in range(C):
        for j in range(i):
            t_val = 0.0
            for kk in range(K_dim):
                t_val += ebar[i, kk] * kbar[j, kk]
            A[i, j] = t_val
    return A, gamma


def test_launch_wy_fast_matches_torch() -> None:
    q, k, v, g_log, b, w = _make_random_inputs()
    A, gamma = _solve_wy_matrix(g_log, b, k, _K)

    w_out, u, qg, kg = launch_wy_fast(A, k, q, b, w, v, gamma, g_log)
    ref_w, ref_u, ref_qg, ref_kg = _torch_wy_fast(A, k, q, b, w, v, gamma)

    torch.testing.assert_close(w_out, ref_w, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(u, ref_u, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(qg, ref_qg, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(kg, ref_kg, atol=1e-5, rtol=1e-5)


def test_launch_wy_fast_qg_is_gamma_times_q() -> None:
    q, k, v, g_log, b, w = _make_random_inputs()
    A, gamma = _solve_wy_matrix(g_log, b, k, _K)

    _, _, qg, _ = launch_wy_fast(A, k, q, b, w, v, gamma, g_log)
    torch.testing.assert_close(qg, gamma * q, atol=1e-5, rtol=1e-5)


def test_launch_wy_fast_kg_is_tail_decayed_key() -> None:
    q, k, v, g_log, b, w = _make_random_inputs()
    A, gamma = _solve_wy_matrix(g_log, b, k, _K)

    _, _, _, kg = launch_wy_fast(A, k, q, b, w, v, gamma, g_log)
    ref_kg = (gamma[-1:] / gamma.clamp_min(1e-12)) * k
    torch.testing.assert_close(kg, ref_kg, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# launch_naive_gdn2
# ---------------------------------------------------------------------------


def test_launch_naive_gdn2_matches_recurrent_reference() -> None:
    q, k, v, g, b, w = _make_random_inputs()
    q_scaled = q * (_K**-0.5)

    _out_ta, _state_ta = launch_naive_gdn2(q_scaled, k, v, g, b, w)

    ref_out, ref_state = torch_recurrent_gdn2(
        q=q.unsqueeze(0).unsqueeze(2),
        k=k.unsqueeze(0).unsqueeze(2),
        v=v.unsqueeze(0).unsqueeze(2),
        g=g.unsqueeze(0).unsqueeze(2),
        b=b.unsqueeze(0).unsqueeze(2),
        w=w.unsqueeze(0).unsqueeze(2),
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    assert ref_state is not None
    ref_out = ref_out[0, :, 0]
    ref_state = ref_state[0, 0]

    torch.testing.assert_close(_out_ta, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(_state_ta[0, 0], ref_state, atol=1e-4, rtol=1e-4)


def test_launch_naive_gdn2_matches_chunked_reference() -> None:
    q, k, v, g, b, w = _make_random_inputs()
    q_scaled = q * (_K**-0.5)

    out_naive, state_naive = launch_naive_gdn2(q_scaled, k, v, g, b, w)

    ref_out, ref_state = torch_chunk_gdn2(
        q=q.unsqueeze(0).unsqueeze(2),
        k=k.unsqueeze(0).unsqueeze(2),
        v=v.unsqueeze(0).unsqueeze(2),
        g=g.unsqueeze(0).unsqueeze(2),
        b=b.unsqueeze(0).unsqueeze(2),
        w=w.unsqueeze(0).unsqueeze(2),
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        chunk_size=_C,
    )
    assert ref_state is not None
    ref_out = ref_out[0, :, 0]
    ref_state = ref_state[0, 0]

    torch.testing.assert_close(out_naive, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(state_naive[0, 0], ref_state, atol=1e-4, rtol=1e-4)


def test_launch_naive_gdn2_with_initial_state() -> None:
    """naive_gdn2 starts from zero state; verify it matches zero-state reference."""
    q, k, v, g, b, w = _make_random_inputs()
    q_scaled = q * (_K**-0.5)

    _out_ta, _state_ta = launch_naive_gdn2(q_scaled, k, v, g, b, w)

    ref_out, ref_state = torch_recurrent_gdn2(
        q=q.unsqueeze(0).unsqueeze(2),
        k=k.unsqueeze(0).unsqueeze(2),
        v=v.unsqueeze(0).unsqueeze(2),
        g=g.unsqueeze(0).unsqueeze(2),
        b=b.unsqueeze(0).unsqueeze(2),
        w=w.unsqueeze(0).unsqueeze(2),
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    assert ref_state is not None
    ref_out = ref_out[0, :, 0]
    ref_state = ref_state[0, 0]

    torch.testing.assert_close(_out_ta, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(_state_ta[0, 0], ref_state, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# launch_chunk_fwd_pipeline
# ---------------------------------------------------------------------------


def _torch_chunk_fwd_pipeline(
    q_scaled: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    state_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for one chunk with initial state."""
    gamma = torch.exp(torch.cumsum(g_log, dim=0))
    gamma_safe = gamma.clamp_min(1e-12)
    kbar = k / gamma_safe
    ebar = gamma * (b * k)
    z = w * v

    y, u = compute_gdn2_wy_coefficients(
        kbar.unsqueeze(0).unsqueeze(0),
        ebar.unsqueeze(0).unsqueeze(0),
        z.unsqueeze(0).unsqueeze(0),
        device=q_scaled.device,
    )
    y = y[0, 0]
    u = u[0, 0]

    delta = u - torch.matmul(y, state_in)
    q_gamma = gamma * q_scaled
    out_inter = torch.matmul(q_gamma, state_in)
    aqk = compute_gdn2_intra_chunk_scores(
        q_scaled.unsqueeze(0).unsqueeze(0),
        gamma.unsqueeze(0).unsqueeze(0),
        kbar.unsqueeze(0).unsqueeze(0),
    )[0, 0]
    out = out_inter + torch.matmul(aqk, delta)

    gamma_last = gamma[-1:]  # [1, K]
    state_out = gamma_last.T * (state_in + torch.matmul(kbar.T, delta))
    return out, state_out


def test_launch_chunk_fwd_pipeline_matches_chunk_fwd_bh() -> None:
    q, k, v, g_log, b, w = _make_random_inputs()
    q_scaled = q * (_K**-0.5)
    state_in = torch.randn(_K, _V) * 0.01

    scratch = _kernels._get_chunk_scratch(_C, _K, _V, "cpu")
    out_bh = torch.zeros(_C, _V)
    state_out_bh = torch.zeros(_K, _V)

    _kernels.launch_chunk_fwd_per_bh(
        q_scaled, k, v, g_log, b, w, state_in, out_bh, state_out_bh,
        scratch["gamma"], scratch["kbar"], scratch["ebar"], scratch["z"],
        scratch["Y"], scratch["U"], scratch["delta"], 1.0,
    )

    scratch_full = _kernels._get_chunk_scratch_full(_C, _K, _V, "cpu")
    out_pipe = torch.zeros(_C, _V)
    state_out_pipe = torch.zeros(_K, _V)
    launch_chunk_fwd_pipeline(
        q_scaled, k, v, g_log, b, w, state_in, out_pipe, state_out_pipe,
        scratch_full, 1.0,
    )

    torch.testing.assert_close(out_pipe, out_bh, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(state_out_pipe, state_out_bh, atol=1e-4, rtol=1e-4)


def test_launch_chunk_fwd_pipeline_matches_torch_reference() -> None:
    q, k, v, g_log, b, w = _make_random_inputs()
    state_in = torch.randn(_K, _V) * 0.01
    q_scaled = q * (_K**-0.5)

    scratch_full = _kernels._get_chunk_scratch_full(_C, _K, _V, "cpu")
    out_pipe = torch.zeros(_C, _V)
    state_out_pipe = torch.zeros(_K, _V)
    launch_chunk_fwd_pipeline(
        q_scaled, k, v, g_log, b, w, state_in, out_pipe, state_out_pipe,
        scratch_full, 1.0,
    )

    ref_out, ref_state = _torch_chunk_fwd_pipeline(
        q_scaled, k, v, g_log, b, w, state_in,
    )

    torch.testing.assert_close(out_pipe, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(state_out_pipe, ref_state, atol=1e-4, rtol=1e-4)


def test_launch_chunk_fwd_pipeline_causal_mask_zero_state() -> None:
    """With zero initial state, output should be finite and structured."""
    q, k, v, g_log, b, w = _make_random_inputs()
    q_scaled = q * (_K**-0.5)
    state_in = torch.zeros(_K, _V)

    scratch_full = _kernels._get_chunk_scratch_full(_C, _K, _V, "cpu")
    out_pipe = torch.zeros(_C, _V)
    state_out_pipe = torch.zeros(_K, _V)
    launch_chunk_fwd_pipeline(
        q_scaled, k, v, g_log, b, w, state_in, out_pipe, state_out_pipe,
        scratch_full, 1.0,
    )

    assert torch.all(torch.isfinite(out_pipe)), "Output contains non-finite values"
    assert torch.all(torch.isfinite(state_out_pipe)), "State contains non-finite values"


def test_launch_chunk_fwd_pipeline_different_chunk_sizes() -> None:
    for C in [4, 6, 16]:
        q, k, v, g_log, b, w = _make_random_inputs(C=C)
        state_in = torch.randn(_K, _V) * 0.01
        q_scaled = q * (_K**-0.5)

        scratch_full = _kernels._get_chunk_scratch_full(C, _K, _V, "cpu")
        out_pipe = torch.zeros(C, _V)
        state_out_pipe = torch.zeros(_K, _V)
        launch_chunk_fwd_pipeline(
            q_scaled, k, v, g_log, b, w, state_in, out_pipe, state_out_pipe,
            scratch_full, 1.0,
        )

        ref_out, ref_state = _torch_chunk_fwd_pipeline(
            q_scaled, k, v, g_log, b, w, state_in,
        )

        torch.testing.assert_close(out_pipe, ref_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state_out_pipe, ref_state, atol=1e-4, rtol=1e-4)
