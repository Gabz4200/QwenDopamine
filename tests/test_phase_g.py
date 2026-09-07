"""Phase G tests: N2 perplexity, N3 warmup, N4 move_to_device, N11 pyproject, N12 transformer."""

from __future__ import annotations

from typing import NamedTuple

import pytest


# N2 ---


def test_perplexity_raises_on_zero_tokens() -> None:
    """Review N2: total_tokens == 0 must raise ValueError, not silently
    return ppl=1.0.
    """
    from unittest.mock import MagicMock

    from qwendopamine.evaluation import perplexity

    model = MagicMock()
    # Empty dataloader -> no iterations -> total_tokens == 0.
    dataloader: list = []
    with pytest.raises(ValueError, match="total_tokens == 0"):
        perplexity.compute_perplexity(model, dataloader, max_steps=10)


# N3 ---


def test_load_state_dict_does_not_reset_lr_when_step_count_zero() -> None:
    """Review N3: a fresh load with step_count=0 must not re-apply warmup
    (which would zero out the LR).
    """
    import torch

    from qwendopamine.training.schedules import LinearWarmupScheduler

    p = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.SGD([p], lr=0.1)
    base = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=0.01)
    sched = LinearWarmupScheduler(optim, base, warmup_steps=10, min_lr=0.01)
    # Advance to step 5 (mid-warmup).
    for _ in range(5):
        sched.step()
    state = sched.state_dict()
    # Build a fresh scheduler and load the state.
    p2 = torch.nn.Parameter(torch.zeros(1))
    optim2 = torch.optim.SGD([p2], lr=0.1)
    base2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=100, eta_min=0.01)
    sched2 = LinearWarmupScheduler(optim2, base2, warmup_steps=10, min_lr=0.01)
    # Force step_count = 0 (simulate a fresh restore).
    state["step_count"] = 0
    sched2.load_state_dict(state)
    # LR must NOT be 0.0 (the bug). It should reflect the base scheduler's
    # eta_min (0.01) or its natural start.
    lr = optim2.param_groups[0]["lr"]
    assert lr > 0, f"load_state_dict reset LR to {lr}; review N3."


# N4 ---


def test_move_to_device_handles_namedtuple() -> None:
    """Review N4: NamedTuple containers must be preserved with tensors moved."""

    class _Batch(NamedTuple):
        x: "object"
        y: "object"

    from qwendopamine.utils import move_to_device

    t1 = __import__("torch").tensor([1.0, 2.0])
    t2 = __import__("torch").tensor([3.0, 4.0])
    batch = _Batch(x=t1, y=t2)
    moved = move_to_device(batch, __import__("torch").device("cpu"))
    assert isinstance(moved, _Batch), (
        f"move_to_device must preserve NamedTuple type; got {type(moved)}"
    )
    assert moved.x.device.type == "cpu"
    assert moved.y.device.type == "cpu"


def test_move_to_device_handles_frozenset() -> None:
    """Review N4: frozenset must be handled (it was previously ignored)."""
    from qwendopamine.utils import move_to_device

    t1 = __import__("torch").tensor([1.0])
    s = frozenset([t1])
    moved = move_to_device(s, __import__("torch").device("cpu"))
    assert isinstance(moved, frozenset)
    assert len(moved) == 1


# N11 ---


def test_pyproject_parsed_not_asserted_as_text() -> None:
    """Review N11: the dependency test must parse pyproject.toml with
    tomllib rather than asserting raw text.
    """
    import inspect
    import sys

    mod = sys.modules.get("tests.test_dependency_compatibility")
    if mod is None:
        # pytest may not have imported the module under this name; load
        # by path instead.
        from importlib.util import spec_from_file_location, module_from_spec

        spec = spec_from_file_location(
            "tests.test_dependency_compatibility",
            "tests/test_dependency_compatibility.py",
        )
        mod = module_from_spec(spec)  # type: ignore[arg-type]
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)
    src = inspect.getsource(mod)
    assert "tomllib" in src or "tomli" in src, (
        "test_dependency_compatibility must use tomllib/tomli to parse "
        "pyproject.toml (review N11)."
    )


# N12 ---


def test_transformer_property_caches_module_dict() -> None:
    """Review N12: model.transformer must return the same object on every access."""
    from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig
    from qwendopamine.models.gdn2_gpt.model import GDN2GPT

    cfg = GDN2GPTConfig(
        n_layer=1,
        n_head=1,
        n_query_groups=1,
        n_embd=8,
        head_size=8,
        block_size=4,
        mlp=False,
    )
    if not hasattr(cfg, "qk_norm"):
        object.__setattr__(cfg, "qk_norm", False)
    model = GDN2GPT(cfg)
    t1 = model.transformer
    t2 = model.transformer
    assert t1 is t2, (
        f"transformer property must return the same object on every call "
        f"(review N12); got two distinct objects: {id(t1)} vs {id(t2)}"
    )
