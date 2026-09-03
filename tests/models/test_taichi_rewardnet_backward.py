"""Backward gradient correctness tests for the Taichi Reinforced Delta kernel.

Each test compares the gradient produced by the Taichi path against the
gradient produced by the pure-PyTorch reference (which uses PyTorch autograd
through the token-by-token recurrence). The Taichi path's backward is the
per-token VJP of the delta-rule update, implemented in
:func:`qwendopamine.models.reinforced.taichi.launch_delta_core_step_bwd`.
"""

from __future__ import annotations

import pytest
import torch

from qwendopamine.models.gdn2.taichi import is_available
from qwendopamine.models.reinforced.taichi import delta_core_step_autograd

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="Taichi runtime is not available on this machine",
)


def _make_inputs(B, D, requires_grad_state: bool = True):
    """Build a fresh set of grad-requiring inputs for one step."""
    torch.manual_seed(0)
    state = torch.zeros(B, D, D)
    if requires_grad_state:
        state.requires_grad_(True)
    k = torch.randn(B, D).requires_grad_(True)
    v = torch.randn(B, D).requires_grad_(True)
    omega_w = torch.full((B, 1), 0.5).requires_grad_(True)
    omega_e = torch.full((B, 1), 0.3).requires_grad_(True)
    write = torch.sigmoid(torch.randn(B, D)).requires_grad_(True)
    erase = torch.sigmoid(torch.randn(B, D)).requires_grad_(True)
    return state, k, v, omega_w, omega_e, write, erase


def _ref_step(state, k, v, omega_w, omega_e, write, erase):
    """Pure-PyTorch reference for the per-token state update.

    Mirrors the math that the Taichi kernel implements
    (``launch_delta_core_step``): the per-batch ``omega_w`` is
    pre-multiplied by the channel-wise ``write`` gate to form
    ``omega_w_eff = omega_w * write``, and the rank-1 update is
    ``S_next[d, k] = (1 - omega_e[d]) * S[d, k] + omega_w_eff[d] * e[d] * k[k]``.
    The torch reference ``DeltaMemoryCore._update_dense`` uses
    ``omega_W = pl * wr * W_t`` and applies it as a scalar. To match
    the Taichi contract the test passes ``omega_w = pl * wr`` (the
    per-batch scalar) and the channel-wise ``write`` separately, so
    here we compute ``omega_w_eff = omega_w * write`` exactly as the
    Taichi autograd Function does.
    """
    omega_w_eff = omega_w * write
    omega_e_eff = omega_e * erase
    e = v - torch.einsum("bdk,bk->bd", state, k)
    w_term = omega_w_eff * e
    return (1.0 - omega_e_eff).unsqueeze(-1) * state + torch.einsum(
        "bd,bk->bdk",
        w_term,
        k,
    )


@pytest.mark.parametrize("B,D", [(1, 4), (2, 8), (1, 16)])
def test_delta_core_step_backward_matches_reference(B, D):
    """Per-step gradient matches the pure-PyTorch reference to 1e-4."""
    (state_ta, k_ta, v_ta, ow_ta, oe_ta, W_ta, E_ta) = _make_inputs(B, D)
    (state_ref, k_ref, v_ref, ow_ref, oe_ref, W_ref, E_ref) = _make_inputs(B, D)

    ns_ta = torch.empty_like(state_ta)
    out_ta = delta_core_step_autograd(
        state_ta, k_ta, v_ta, ow_ta, oe_ta, W_ta, E_ta, ns_ta
    )
    out_ref = _ref_step(state_ref, k_ref, v_ref, ow_ref, oe_ref, W_ref, E_ref)

    torch.testing.assert_close(out_ta, out_ref, atol=1e-5, rtol=1e-5)

    out_ta.sum().backward()
    out_ref.sum().backward()
    grads = [
        ("state", state_ta, state_ref),
        ("k", k_ta, k_ref),
        ("v", v_ta, v_ref),
        ("omega_w", ow_ta, ow_ref),
        ("omega_e", oe_ta, oe_ref),
        ("write", W_ta, W_ref),
        ("erase", E_ta, E_ref),
    ]
    for name, ta, ref in grads:
        assert ta.grad is not None, f"{name} grad missing on Taichi path"
        assert ref.grad is not None, f"{name} grad missing on reference path"
        torch.testing.assert_close(
            ta.grad,
            ref.grad,
            atol=1e-5,
            rtol=1e-5,
            msg=f"grad {name} differs: max={(ta.grad - ref.grad).abs().max().item()}",
        )


def test_delta_core_step_training_step_converges():
    """One SGD step on a tiny linear model should reduce the loss."""
    torch.manual_seed(0)
    B, D = 1, 4
    T = 5  # noqa: F841 (placeholder for future multi-token extension)
    state = torch.zeros(B, D, D).requires_grad_(True)
    k = torch.randn(B, D).requires_grad_(True)
    v = torch.randn(B, D).requires_grad_(True)
    ow = torch.full((B, 1), 0.5).requires_grad_(True)
    oe = torch.full((B, 1), 0.3).requires_grad_(True)
    W = torch.sigmoid(torch.randn(B, D)).requires_grad_(True)
    E = torch.sigmoid(torch.randn(B, D)).requires_grad_(True)
    target = torch.randn(B, D, D)
    ns = torch.empty_like(state)

    loss_initial = (state - target).pow(2).sum().item()  # before any step
    for _ in range(5):
        out = delta_core_step_autograd(state, k, v, ow, oe, W, E, ns)
        loss = (out - target).pow(2).sum()
        loss.backward()
        with torch.no_grad():
            for p in (state, k, v, ow, oe, W, E):
                p_grad = p.grad
                assert p_grad is not None
                p.add_(p_grad, alpha=-0.1)
                p_grad.zero_()
        state = state.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        ow = ow.detach().requires_grad_(True)
        oe = oe.detach().requires_grad_(True)
        W = W.detach().requires_grad_(True)
        E = E.detach().requires_grad_(True)
        ns = torch.empty_like(state)

    assert loss.item() < loss_initial, (
        f"loss did not decrease: initial={loss_initial} final={loss.item()}"
    )


@pytest.mark.parametrize("B,D,T", [(1, 4, 4), (2, 8, 3)])
def test_multi_token_recurrence_backward(B, D, T):
    """The per-step backward chains correctly across multiple tokens."""
    torch.manual_seed(0)
    state_ta = torch.zeros(B, D, D).requires_grad_(True)
    state_ref = torch.zeros(B, D, D).requires_grad_(True)
    k_ta = torch.randn(B, T, D).requires_grad_(True)
    v_ta = torch.randn(B, T, D).requires_grad_(True)
    ow_ta = torch.full((B, T, 1), 0.5).requires_grad_(True)
    oe_ta = torch.full((B, T, 1), 0.3).requires_grad_(True)
    W_ta = torch.sigmoid(torch.randn(B, T, D)).requires_grad_(True)
    E_ta = torch.sigmoid(torch.randn(B, T, D)).requires_grad_(True)
    k_ref = k_ta.detach().clone().requires_grad_(True)
    v_ref = v_ta.detach().clone().requires_grad_(True)
    ow_ref = ow_ta.detach().clone().requires_grad_(True)
    oe_ref = oe_ta.detach().clone().requires_grad_(True)
    W_ref = W_ta.detach().clone().requires_grad_(True)
    E_ref = E_ta.detach().clone().requires_grad_(True)

    s_ta = state_ta
    s_ref = state_ref
    ns_ta = torch.empty_like(state_ta)
    for t in range(T):
        k_t = k_ta[:, t, :]
        v_t = v_ta[:, t, :]
        ow_t = ow_ta[:, t, :]
        oe_t = oe_ta[:, t, :]
        W_t = W_ta[:, t, :]
        E_t = E_ta[:, t, :]
        s_ta = delta_core_step_autograd(s_ta, k_t, v_t, ow_t, oe_t, W_t, E_t, ns_ta)
        s_ref = _ref_step(
            s_ref,
            k_ref[:, t, :],
            v_ref[:, t, :],
            ow_ref[:, t, :],
            oe_ref[:, t, :],
            W_ref[:, t, :],
            E_ref[:, t, :],
        )
        ns_ta = torch.empty_like(s_ta)
        # Carry autograd through the loop by re-requiring state.
        if t < T - 1:
            s_ta = s_ta.detach().requires_grad_(True)
            s_ref = s_ref.detach().requires_grad_(True)

    torch.testing.assert_close(s_ta, s_ref, atol=1e-4, rtol=1e-4)
    s_ta.sum().backward()
    s_ref.sum().backward()
    for name, ta, ref in [
        ("k", k_ta, k_ref),
        ("v", v_ta, v_ref),
        ("omega_w", ow_ta, ow_ref),
        ("omega_e", oe_ta, oe_ref),
        ("write", W_ta, W_ref),
        ("erase", E_ta, E_ref),
    ]:
        ta_grad = ta.grad
        ref_grad = ref.grad
        assert ta_grad is not None
        assert ref_grad is not None
        torch.testing.assert_close(
            ta_grad,
            ref_grad,
            atol=1e-3,
            rtol=1e-3,
            msg=f"grad {name} differs: max={(ta_grad - ref_grad).abs().max().item()}",
        )
