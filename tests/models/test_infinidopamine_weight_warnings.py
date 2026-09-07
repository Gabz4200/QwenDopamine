"""Tests for M7 + N1: mtp.* backfill warning, drop unused load_info."""

from __future__ import annotations

import logging

import pytest


def test_load_info_list_is_removed() -> None:
    """N1: the unused ``load_info`` list must no longer be constructed.

    The previous build of the multimodal loader collected a
    ``load_info`` list that was never returned, logged, or surfaced.
    The fix (review N1) replaces it with structured per-partition INFO
    logs and removes the list entirely.
    """
    import inspect

    from qwendopamine.models.infinidopamine import _qwen35_weights as m

    src = inspect.getsource(m)
    assert "load_info: list[str] = []" not in src, (
        f"Found unused 'load_info' list still being constructed in "
        f"{m.__name__}. Review N1: replace with logging or drop the list."
    )


def test_mtp_keys_emit_warning(caplog: pytest.LogCaptureFixture) -> None:
    """M7: the multimodal loader must warn when dropping mtp.* keys.

    Previously the loop did ``continue`` for any key starting with
    ``mtp.``, leaving the user with no signal. The fix logs at WARNING
    with the count and a sample of the dropped keys.
    """
    import torch

    from qwendopamine.models.infinidopamine import _qwen35_weights as m

    class _StubVisual(torch.nn.Module):
        def load_state_dict(self, sd, strict: bool = True):  # type: ignore[override]
            return [], []

    class _StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = _StubVisual()
            self.language_model = _StubLang()

    class _StubLang(torch.nn.Module):
        def load_qwen35_weights(self, sd, strict: bool = True):  # type: ignore[override]
            return [], []

        def load_state_dict(self, sd, strict: bool = True):  # type: ignore[override]
            return [], []

    class _StubTop(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _StubModel()
            self.lm_head = torch.nn.Linear(2, 2)

    top = _StubTop()
    # Construct a state dict with three mtp.* keys plus a normal key.
    state = {
        "model.language_model.fc.weight": torch.zeros(2, 2),
        "mtp.fc1.weight": torch.zeros(2, 2),
        "mtp.fc2.weight": torch.zeros(2, 2),
        "mtp.fc3.weight": torch.zeros(2, 2),
    }
    with caplog.at_level(logging.WARNING, logger=m.__name__):
        result = m.load_qwen35_weights(top, state, strict=True)
    assert result is not None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("mtp" in r.message for r in warnings), (
        f"Expected a WARNING about mtp.* drops; got {[r.message for r in warnings]!r}"
    )
    # The warning should report the count (3).
    assert any("3" in r.message for r in warnings), (
        f"Expected the mtp drop count (3) in the warning; got {[r.message for r in warnings]!r}"
    )


def test_text_loader_warns_on_mtp(caplog: pytest.LogCaptureFixture) -> None:
    """M7 (text variant): the text loader must also warn on mtp.* drops."""
    import torch

    from qwendopamine.models.infinidopamine import _text_qwen35_weights as tm

    class _Stub(torch.nn.Module):
        def load_state_dict(self, sd, strict: bool = True):  # type: ignore[override]
            return [], []

    model = _Stub()
    state = {
        "model.language_model.fc.weight": torch.zeros(2, 2),
        "mtp.foo.weight": torch.zeros(2, 2),
    }
    with caplog.at_level(logging.WARNING, logger=tm.__name__):
        tm.load_text_qwen35_weights(model, state, strict=True)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("mtp" in r.message for r in warnings), (
        f"Expected WARNING about mtp.*; got {[r.message for r in warnings]!r}"
    )
