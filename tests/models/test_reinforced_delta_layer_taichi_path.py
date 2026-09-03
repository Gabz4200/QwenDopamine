"""Test that ``ReinforcedDeltaLayer`` uses the Taichi autograd path when
``use_taichi=True`` (the default) and that the Taichi path agrees with
the pure-PyTorch path for the dense memory case.

Skipped when the Taichi runtime is not available; the pure-PyTorch
fallback is exercised in :mod:`tests.models.test_reward`.
"""

from __future__ import annotations

import pytest
import torch

from qwendopamine.models.gdn2.taichi import is_available
from qwendopamine.models.reinforced import (
    GatedRewardNet,
    GatedRewardNetConfig,
)
from qwendopamine.models.reinforced.delta import ReinforcedDeltaLayer
from qwendopamine.models.reinforced.taichi import _DeltaCoreStepFunction

taichi_skip = pytest.mark.skipif(
    not is_available(), reason="Taichi runtime not available"
)


def _make_layer(*, use_taichi: bool, d_model: int = 16) -> ReinforcedDeltaLayer:
    """Build a small ``ReinforcedDeltaLayer`` with the given Taichi flag.

    Uses the default FiLM reward encoder (which expects the per-step
    shape ``(B, k_stats)`` the layer feeds it).
    """
    from qwendopamine.models.reinforced.delta import _DefaultQueryFiLM

    reward_encoder = _DefaultQueryFiLM(k=6, d=d_model)
    return ReinforcedDeltaLayer(
        d_model=d_model,
        k_stats=6,
        reward_encoder=reward_encoder,
        use_short_conv=False,
        use_taichi=use_taichi,
    )


@taichi_skip
def test_use_taichi_default_invokes_taichi_function() -> None:
    """``use_taichi=True`` (default) routes the per-step state update
    through the Taichi autograd Function so the chain of ``grad_fn``
    contains ``_DeltaCoreStepFunction``.
    """
    layer = _make_layer(use_taichi=True, d_model=8)
    x = torch.randn(2, 8)
    reward_values = torch.randn(2, 6)
    _, S_next, _, _, _ = layer(x, reward_values=reward_values)
    # When the forward went through the Taichi kernel, the autograd
    # graph attached to ``S_next`` ends at ``_DeltaCoreStepFunctionBackward``
    # (PyTorch wraps the custom Function's backward in a private
    # ``*Backward`` shim during the first backward pass — and that
    # shim is the grad_fn actually stored on the saved tensor at
    # forward time, because the Function's ``backward`` is registered
    # eagerly). We therefore accept either form.
    found = _uses_taichi_grad_fn(S_next)
    assert found, (
        "use_taichi=True did not route through the Taichi autograd "
        f"Function; got grad_fn={S_next.grad_fn!r}"
    )


@taichi_skip
def test_torch_and_taichi_paths_match() -> None:
    """Two ``ReinforcedDeltaLayer`` instances built with the same seed,
    one with ``use_taichi=False`` and one with ``use_taichi=True``,
    produce the same output and final state for an identical input
    and reward stream.
    """
    torch.manual_seed(0)
    layer_torch = _make_layer(use_taichi=False, d_model=8)
    torch.manual_seed(0)
    layer_taichi = _make_layer(use_taichi=True, d_model=8)

    # Same input and reward stream.
    torch.manual_seed(1)
    x = torch.randn(2, 8)
    reward_values = torch.randn(2, 6)

    o_torch, S_torch, V_torch, _, _ = layer_torch(
        x,
        reward_values=reward_values,
    )
    o_taichi, S_taichi, V_taichi, _, _ = layer_taichi(
        x,
        reward_values=reward_values,
    )

    torch.testing.assert_close(
        o_taichi,
        o_torch,
        atol=1e-5,
        rtol=1e-5,
        msg="Output features differ between Taichi and torch paths.",
    )
    torch.testing.assert_close(
        S_taichi,
        S_torch,
        atol=1e-5,
        rtol=1e-5,
        msg="Final state differs between Taichi and torch paths.",
    )
    torch.testing.assert_close(
        V_taichi,
        V_torch,
        atol=1e-5,
        rtol=1e-5,
        msg="EMA baseline differs between Taichi and torch paths.",
    )


@taichi_skip
def test_one_sgd_step_decreases_loss() -> None:
    """One SGD step on a small ``ReinforcedDeltaLayer`` (Taichi path)
    decreases the MSE against a fixed target.
    """
    torch.manual_seed(2)
    layer = _make_layer(use_taichi=True, d_model=8)
    opt = torch.optim.SGD(layer.parameters(), lr=0.01)
    x = torch.randn(2, 8)
    reward_values = torch.randn(2, 6)
    target = torch.randn(2, 8)

    opt.zero_grad()
    o, _, _, _, _ = layer(x, reward_values=reward_values)
    loss = (o - target).pow(2).mean()
    initial_loss = loss.item()
    loss.backward()
    opt.step()

    opt.zero_grad()
    o, _, _, _, _ = layer(x, reward_values=reward_values)
    new_loss = (o - target).pow(2).mean().item()
    assert new_loss < initial_loss, (
        f"Expected one SGD step to decrease loss, "
        f"got initial={initial_loss} new={new_loss}"
    )


def test_use_taichi_disabled_falls_back_to_torch() -> None:
    """``use_taichi=False`` keeps the pure-PyTorch step; the state update
    does not go through the Taichi autograd Function.
    """
    layer = _make_layer(use_taichi=False, d_model=8)
    x = torch.randn(2, 8)
    reward_values = torch.randn(2, 6)
    _, S_next, _, _, _ = layer(x, reward_values=reward_values)
    assert S_next.grad_fn is None or not _uses_taichi_grad_fn(S_next)


def _uses_taichi_grad_fn(t: torch.Tensor) -> bool:
    """Return True if any node in ``t.grad_fn`` chain is the Taichi
    autograd Function (or its private ``*Backward`` shim).
    """
    grad_fn = t.grad_fn
    while grad_fn is not None:
        cls_name = type(grad_fn).__name__
        if (
            isinstance(grad_fn, _DeltaCoreStepFunction)
            or cls_name == "_DeltaCoreStepFunctionBackward"
            or cls_name.endswith("Backward")
            and "DeltaCoreStepFunction" in cls_name
        ):
            return True
        next_fns = getattr(grad_fn, "next_functions", None)
        if not next_fns:
            return False
        grad_fn = next_fns[0][0]
    return False


@taichi_skip
def test_gated_reward_net_smoke_with_taichi_path() -> None:
    """``GatedRewardNet`` with default Taichi path runs a 4-token
    sequence and produces finite outputs.
    """
    grn = GatedRewardNet(
        GatedRewardNetConfig(
            hidden_size=16,
            k_stats=6,
            use_short_conv=False,
        )
    )
    grn.eval()
    inputs = torch.randn(1, 4, 16)
    rewards = torch.randn(1, 4, 6)
    out, _, _ = grn(inputs, reward_values=rewards, use_cache=True)
    assert out.shape == (1, 4, 16)
    assert torch.isfinite(out).all()
