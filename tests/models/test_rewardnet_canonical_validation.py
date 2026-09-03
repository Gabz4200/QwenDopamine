"""Third-reference validation tests for the RewardNet forward and backward.

Each test compares the per-step and per-sequence RewardNet update
against a hand-derived third reference that is **independent** of
both the local torch reference
(:mod:`qwendopamine.models.reinforced.delta`) and the Taichi kernels
(:mod:`qwendopamine.kernels.taichi.reinforced_kernels`). Tests are skipped when
the Taichi runtime is not available; the pure-PyTorch path is still
exercised via :mod:`tests.models.test_reward`.

Contract (single-step):
    state          [B, D, D]
    k, v           [B, D]
    omega_w        [B, 1]  (per-batch scalar)        \
    write          [B, D]  (per-channel gate)        |  -> omega_w_eff [B, D]
    omega_e        [B, 1]  (per-batch scalar)        |  -> omega_e_eff [B, D]
    erase          [B, D]  (per-channel gate)        /

The canonical accepts pre-multiplied ``[B, D]`` per-channel effective
gates directly. The torch autograd and the Taichi autograd Function
both compose ``omega_w_eff = omega_w * write`` internally and then
distribute gradients back to the original inputs without division.
"""

from __future__ import annotations

import pytest
import torch

from qwendopamine.kernels.taichi import is_available
from qwendopamine.models.reinforced import ReinforcedDeltaLayer
from qwendopamine.models.reinforced.canonical_reference import (
    canonical_delta_step,
    canonical_delta_step_with_grad,
)
from qwendopamine.models.reinforced.delta import _DefaultQueryFiLM

# Tolerance for fp32 accumulation noise (no L2 norm, no scale in canonical).
_F32_ATOL = 1e-5
_F32_RTOL = 1e-5

# Shapes including non-square D to catch axis confusion.
SHAPES_SINGLE = [(1, 4), (2, 6), (1, 8)]
SHAPES_SEQ = [(1, 4), (2, 6), (1, 8)]


def _rand_inputs(B: int, D: int, *, seed: int = 0):
    """Random per-step inputs (state, k, v) + per-channel effective gates.

    The canonical reference operates on the pre-multiplied per-channel
    effective gates ``omega_w_eff`` and ``omega_e_eff`` of shape
    ``[B, D]``. Tests that exercise the torch / Taichi autograd
    rebuild the per-batch scalar ``omega_w`` and per-channel ``write``
    from these gates by using ``write = ones`` and a constant scalar
    multiplier (see e.g. ``test_torch_per_step_grad_matches_canonical``).

    Both effective gates are constant per-channel so the Taichi
    autograd path (which composes ``omega_w * write`` with
    ``write = ones``) receives the same per-channel buffer as the
    canonical.
    """
    torch.manual_seed(seed)
    S = torch.randn(B, D, D)
    k = torch.randn(B, D) * 0.5
    v = torch.randn(B, D) * 0.5
    omega_w_eff = torch.full((B, D), 0.5)
    omega_e_eff = torch.full((B, D), 0.3)
    return S, k, v, omega_w_eff, omega_e_eff


# ---------------------------------------------------------------------------
# 1. torch single-step vs canonical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,D", SHAPES_SINGLE)
def test_torch_single_step_matches_canonical(B, D):
    """Local torch ``DeltaMemoryCore._update_dense`` matches the canonical step."""
    S, k, v, omega_w_eff, omega_e_eff = _rand_inputs(B, D)
    # The canonical contract: per-channel effective gates [B, D]. The
    # torch ref column-wise update with ``ow_col =
    # omega_w_eff.unsqueeze(-1) = [B, D, 1]`` and ``decay = (1 -
    # omega_e_eff).unsqueeze(-1) = [B, D, 1]`` matches the canonical
    # per-channel update exactly.
    S_canon = canonical_delta_step(S.clone(), k, v, omega_w_eff, omega_e_eff)
    pred = (S @ k.unsqueeze(-1)).squeeze(-1)
    e = v - pred
    outer = torch.bmm(e.unsqueeze(-1), k.unsqueeze(1))
    decay = 1.0 - omega_e_eff.unsqueeze(-1)
    ow_col = omega_w_eff.unsqueeze(-1)
    S_torch = decay * S + ow_col * outer
    torch.testing.assert_close(
        S_canon,
        S_torch,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )


# ---------------------------------------------------------------------------
# 2. per-step VJP: torch autograd vs hand-derived canonical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,D", SHAPES_SINGLE)
def test_torch_per_step_grad_matches_canonical(B, D):
    """Per-step VJP from torch autograd matches the hand-derived one.

    The torch reference uses per-channel ``ow_t`` / ``oe_t`` of shape
    ``[B, D]`` (the same effective gates the canonical expects). The
    autograd grads are then directly comparable to the canonical's
    per-channel gradients.
    """
    S, k, v, omega_w_eff, omega_e_eff = _rand_inputs(B, D)
    dS_next = torch.randn(B, D, D)
    _, dS_c, dk_c, dv_c, d_ow_c, d_oe_c = canonical_delta_step_with_grad(
        S.clone(),
        k,
        v,
        omega_w_eff,
        omega_e_eff,
        dS_next,
    )
    S_t = S.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    ow_t = omega_w_eff.clone().requires_grad_(True)
    oe_t = omega_e_eff.clone().requires_grad_(True)
    pred = S_t @ k_t.unsqueeze(-1)
    e = v_t - pred.squeeze(-1)
    outer = torch.bmm(e.unsqueeze(-1), k_t.unsqueeze(1))
    decay = 1.0 - oe_t.unsqueeze(-1)
    ow_col = ow_t.unsqueeze(-1)
    S_out = decay * S_t + ow_col * outer
    S_out.backward(dS_next)
    torch.testing.assert_close(
        S_t.grad,
        dS_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    torch.testing.assert_close(
        k_t.grad,
        dk_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    torch.testing.assert_close(
        v_t.grad,
        dv_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    torch.testing.assert_close(
        ow_t.grad,
        d_ow_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    torch.testing.assert_close(
        oe_t.grad,
        d_oe_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )


# ---------------------------------------------------------------------------
# 3. sequence vs torch recurrent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,D", SHAPES_SEQ)
def test_torch_sequence_matches_canonical(B, D):
    """Sequential canonical loop matches torch ref loop."""
    T = 3
    S0 = torch.zeros(B, D, D)
    ks = torch.randn(B, T, D) * 0.5
    vs = torch.randn(B, T, D) * 0.5
    # Per-channel effective gates (constant per token) so the test
    # stays isolated from the RewardNet-specific channel-wise
    # plasticity modulation.
    ows = torch.full((B, T, D), 0.5)
    oes = torch.rand(B, T, D) * 0.5
    # Canonical
    S_c = S0.clone()
    for t in range(T):
        S_c = canonical_delta_step(
            S=S_c,
            k_t=ks[:, t, :],
            v_t=vs[:, t, :],
            omega_W=ows[:, t, :],
            omega_E=oes[:, t, :],
        )
    # Torch ref
    S_t = S0.clone()
    for t in range(T):
        pred = S_t @ ks[:, t, :].unsqueeze(-1)
        e = vs[:, t, :] - pred.squeeze(-1)
        outer = torch.bmm(e.unsqueeze(-1), ks[:, t, :].unsqueeze(1))
        decay = 1.0 - oes[:, t, :].unsqueeze(-1)
        ow_col = ows[:, t, :].unsqueeze(-1)
        S_t = decay * S_t + ow_col * outer
    torch.testing.assert_close(
        S_c,
        S_t,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )


# ---------------------------------------------------------------------------
# 4. Taichi per-step forward vs canonical
# ---------------------------------------------------------------------------


def _make_layer_taichi(use_taichi: bool, d_model: int = 8) -> ReinforcedDeltaLayer:
    """Build a tiny ``ReinforcedDeltaLayer`` for the per-step tests."""
    return ReinforcedDeltaLayer(
        d_model=d_model,
        k_stats=6,
        reward_encoder=_DefaultQueryFiLM(k=6, d=d_model),
        use_short_conv=False,
        use_taichi=use_taichi,
    )


@pytest.mark.skipif(not is_available(), reason="Taichi runtime not available")
@pytest.mark.parametrize("B,D", SHAPES_SINGLE)
def test_taichi_per_step_forward_matches_canonical(B, D):
    """Taichi per-step forward matches the canonical single-step.

    Pass per-batch scalar ``omega_w`` / ``omega_e`` of shape ``[B, 1]``
    plus per-channel ``write = ones`` / ``erase = ones`` so the
    effective gates equal the canonical's ``omega_w_eff`` /
    ``omega_e_eff``.
    """
    from qwendopamine.kernels.taichi.reinforced_kernels import launch_delta_core_step

    torch.manual_seed(0)
    S, k, v, omega_w_eff, omega_e_eff = _rand_inputs(B, D, seed=0)
    ow_scalar = torch.full((B, 1), omega_w_eff[0, 0].item())
    oe_scalar = torch.full((B, 1), omega_e_eff[0, 0].item())
    write = torch.ones(B, D)
    erase = torch.ones(B, D)
    S_canon = canonical_delta_step(
        S.clone(),
        k,
        v,
        omega_w_eff,
        omega_e_eff,
    )
    ns = torch.zeros_like(S)
    launch_delta_core_step(
        state=S.float(),
        k=k.float(),
        v=v.float(),
        omega_w=ow_scalar.float(),
        omega_e=oe_scalar.float(),
        erase=erase.float(),
        write=write.float(),
        next_state=ns,
    )
    torch.testing.assert_close(
        ns,
        S_canon,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )


# ---------------------------------------------------------------------------
# 5. Taichi per-step backward vs canonical
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_available(), reason="Taichi runtime not available")
@pytest.mark.parametrize("B,D", SHAPES_SINGLE)
def test_taichi_per_step_backward_matches_canonical(B, D):
    """Taichi per-step VJP matches the canonical hand-derived VJP.

    Composition: ``omega_w_eff = omega_w * write`` (per-channel). The
    Taichi autograd Function distributes gradients back to the
    pre-multiplied inputs without division:

        d_omega_w_eff [B, D]  (from the kernel)
        d_omega_w     [B, 1]  = sum_d d_omega_w_eff * write
        d_write       [B, D]  = d_omega_w_eff * omega_w
        d_omega_e_eff [B, D]  (from the kernel)
        d_omega_e     [B, 1]  = sum_d d_omega_e_eff * erase
        d_erase       [B, D]  = d_omega_e_eff * omega_e
    """
    from qwendopamine.kernels.taichi.reinforced_kernels import delta_core_step_out

    torch.manual_seed(0)
    # Use constant per-channel effective gates so the Taichi autograd
    # path (which composes ``omega_w * write`` and ``omega_e * erase``)
    # matches the canonical's per-channel input exactly.
    omega_w_eff = torch.full((B, D), 0.5)
    omega_e_eff = torch.full((B, D), 0.3)
    S = torch.zeros(B, D, D)
    k = torch.randn(B, D) * 0.5
    v = torch.randn(B, D) * 0.5
    ow_scalar = torch.full((B, 1), 0.5)
    oe_scalar = torch.full((B, 1), 0.3)
    write = torch.ones(B, D)
    erase = torch.ones(B, D)
    dS_next = torch.randn(B, D, D)
    _, dS_c, dk_c, dv_c, d_ow_eff_c, d_oe_eff_c = canonical_delta_step_with_grad(
        S.clone(),
        k,
        v,
        omega_w_eff,
        omega_e_eff,
        dS_next,
    )
    S_t = S.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    v_t = v.clone().requires_grad_(True)
    ow_t = ow_scalar.clone().requires_grad_(True)
    oe_t = oe_scalar.clone().requires_grad_(True)
    write_t = write.clone().requires_grad_(True)
    erase_t = erase.clone().requires_grad_(True)
    ns = torch.empty_like(S_t)
    out = delta_core_step_out(
        S_t,
        k_t,
        v_t,
        ow_t,
        oe_t,
        write_t,
        erase_t,
        ns,
    )
    loss = (out * dS_next).sum()
    loss.backward()
    torch.testing.assert_close(
        S_t.grad,
        dS_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    torch.testing.assert_close(
        k_t.grad,
        dk_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    torch.testing.assert_close(
        v_t.grad,
        dv_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    # d_omega_w [B, 1] = sum_d d_omega_w_eff * write
    d_ow_c_full = (d_ow_eff_c * write).sum(dim=-1, keepdim=True)
    d_oe_c_full = (d_oe_eff_c * erase).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(
        ow_t.grad,
        d_ow_c_full,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    torch.testing.assert_close(
        oe_t.grad,
        d_oe_c_full,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    # d_write [B, D] = d_omega_w_eff * omega_w
    # d_erase [B, D] = d_omega_e_eff * omega_e
    d_write_c = d_ow_eff_c * ow_scalar
    d_erase_c = d_oe_eff_c * oe_scalar
    torch.testing.assert_close(
        write_t.grad,
        d_write_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
    torch.testing.assert_close(
        erase_t.grad,
        d_erase_c,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )


# ---------------------------------------------------------------------------
# 6. Sequence canonical matches torch ref recurrent
# ---------------------------------------------------------------------------


def test_torch_recurrent_sequence_matches_canonical_sequence():
    """``ReinforcedDeltaLayer`` (torch path, no Taichi) recurrent loop
    matches the canonical sequential loop over a 3-token sequence.
    """
    torch.manual_seed(0)
    B, D, T = 1, 8, 3
    layer = _make_layer_taichi(use_taichi=False, d_model=D)
    V_prev = torch.zeros(B, 6)
    S0 = torch.zeros(B, D, D)
    torch.manual_seed(1)
    xs = torch.randn(T, B, D)
    rewards = torch.randn(T, B, 6)
    S_c = S0.clone()
    S_torch = S0.clone()
    for t in range(T):
        x_t = xs[t]
        reward_values = rewards[t]
        with torch.no_grad():
            R = layer.stats_extractor(reward_values, batch_size=B, seq_len=1)
            R = layer.stats_normalizer(R).squeeze(1)
            V_t, A_t = layer.baseline_tracker(x_t, R, V_prev)
            V_prev = V_t
            g = layer.advantage_gate(A_t)
            pl, wr, er = g
            k_t, v_t, W_t, E_t, _, _ = layer.memory_core._compute_step_inputs(x_t)
            ow = pl * wr
            oe = pl * er
            pred = S_torch @ k_t.unsqueeze(-1)
            e = v_t - pred.squeeze(-1)
            outer = torch.bmm(e.unsqueeze(-1), k_t.unsqueeze(1))
            decay = 1.0 - (oe * E_t).unsqueeze(-1)
            ow_eff = (ow * W_t).unsqueeze(-1)
            S_torch = decay * S_torch + ow_eff * outer
            S_c = canonical_delta_step(
                S=S_c,
                k_t=k_t,
                v_t=v_t,
                omega_W=(ow * W_t),
                omega_E=(oe * E_t),
            )
    torch.testing.assert_close(
        S_c,
        S_torch,
        atol=_F32_ATOL,
        rtol=_F32_RTOL,
    )
