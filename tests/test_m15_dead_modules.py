"""M15: dead duplicate modules must be removed."""

from __future__ import annotations

import importlib


def test_trl_module_not_importable() -> None:
    """Review M15: ``qwendopamine.integrations.huggingface.trl`` is a
    dead duplicate of ``_build.py``. After deletion, importing it must
    raise ModuleNotFoundError.
    """
    with __import__("pytest").raises(ModuleNotFoundError):
        importlib.import_module("qwendopamine.integrations.huggingface.trl")


def test_saving_module_not_importable() -> None:
    """Review M15: ``qwendopamine.integrations.huggingface.saving`` is a
    dead duplicate of ``_save.py``. After deletion, importing it must
    raise ModuleNotFoundError.
    """
    with __import__("pytest").raises(ModuleNotFoundError):
        importlib.import_module("qwendopamine.integrations.huggingface.saving")


def test_canonical_helpers_still_importable() -> None:
    """The canonical ``_build`` and ``_save`` modules must remain importable."""
    from qwendopamine.integrations.huggingface import _build, _save

    assert hasattr(_build, "prepare_model_for_trl_training")
    assert _save is not None


def test_fake_impls_module_not_importable() -> None:
    """``fake_impls.py`` was a pure re-export shim of the fakes defined in
    ``custom_ops.py`` with zero importers. After deletion, importing it
    must raise ModuleNotFoundError.
    """
    with __import__("pytest").raises(ModuleNotFoundError):
        importlib.import_module("qwendopamine.integrations.pytorch.fake_impls")


def test_parallel_reward_port_module_not_importable() -> None:
    """``_parallel_reward_port.py`` defined an unused ``ParallelRewardPort``
    protocol. After deletion, importing it must raise ModuleNotFoundError.
    """
    with __import__("pytest").raises(ModuleNotFoundError):
        importlib.import_module(
            "qwendopamine.models.infinidopamine._parallel_reward_port"
        )
