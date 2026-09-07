"""M13: is_autograd_registered / is_registered must reflect actual state.
N8: dead ``if TYPE_CHECKING: pass / else: pass`` stub removed.
"""

from __future__ import annotations

import importlib

import pytest


def test_is_autograd_registered_false_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review M13: ``is_autograd_registered()`` must return False until
    ``register_all_autograd()`` runs. The previous code initialised the
    flag to True at import time, making the check a no-op.
    """
    import qwendopamine.integrations.pytorch.autograd as a

    # Reset the flag to a known initial state.
    monkeypatch.setattr(a, "_REGISTERED_AUTOGRAD", False)
    assert a.is_autograd_registered() is False, (
        "is_autograd_registered() returned True before any call to "
        "register_all_autograd(). Review M13: the flag must track actual state."
    )


def test_is_autograd_registered_true_after_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``register_all_autograd()`` runs, the flag must be True."""
    import qwendopamine.integrations.pytorch.autograd as a

    monkeypatch.setattr(a, "_REGISTERED_AUTOGRAD", False)
    a.register_all_autograd()
    assert a.is_autograd_registered() is True


def test_custom_ops_has_no_type_checking_stub() -> None:
    """N8: the dead ``if TYPE_CHECKING: pass / else: pass`` stub is gone."""
    import inspect

    from qwendopamine.integrations.pytorch import custom_ops

    src = inspect.getsource(custom_ops)
    assert "if TYPE_CHECKING:" not in src, (
        "custom_ops.py still contains the dead 'if TYPE_CHECKING: pass / "
        "else: pass' stub. Review N8: remove it."
    )
    # And the TYPE_CHECKING import is no longer needed.
    assert "TYPE_CHECKING" not in src, (
        "TYPE_CHECKING import is no longer needed after removing the dead stub."
    )


def test_custom_ops_module_importable() -> None:
    """The module must still import cleanly after the stub removal."""
    # Force a re-import to make sure nothing depends on the stub.
    mod = importlib.import_module("qwendopamine.integrations.pytorch.custom_ops")
    assert mod is not None
