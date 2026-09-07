"""N6: replace the AttributeError class-attr sentinel with a clean descriptor."""

from __future__ import annotations


def test_unsupported_attr_descriptor_replaces_sentinel() -> None:
    """The class must NOT contain ``AttributeError()`` sentinels.

    Review N6: ``norm_topk_prob = AttributeError()`` was a fragile
    trick — the AttributeError *instance* sat on the class and was
    returned by ``getattr``, requiring the caller to explicitly raise
    it. The new descriptor raises a clear AttributeError at access
    time.
    """
    import inspect

    from qwendopamine.models.infinidopamine import configs as c

    src_text = inspect.getsource(c)
    assert "= AttributeError()" not in src_text, (
        "configs.py still contains the old '= AttributeError()' "
        "sentinel trick. Review N6: use a clean descriptor."
    )


def test_accessing_unsupported_attr_returns_zero() -> None:
    """Reading an unsupported field returns 0 so parent ``> 0`` checks work.

    The parent ``PretrainedConfig`` machinery compares these fields
    to numbers (e.g. ``num_experts > 0``). The descriptor returns 0 so
    those checks evaluate to "not MoE" without raising.
    """
    from qwendopamine.models.infinidopamine.configs import (
        InfiniDopamineTextConfig,
    )

    cfg = InfiniDopamineTextConfig()
    assert cfg.norm_topk_prob == 0, (
        f"norm_topk_prob must be 0 (sentinel); got {cfg.norm_topk_prob!r}"
    )
