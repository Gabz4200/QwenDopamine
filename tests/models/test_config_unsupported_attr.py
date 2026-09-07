"""N6: replace the AttributeError class-attr sentinel with a clean descriptor."""

from __future__ import annotations

import pytest


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


def test_accessing_unsupported_attr_raises_clear_error() -> None:
    """Reading an unsupported field must raise AttributeError with a useful message."""
    from qwendopamine.models.infinidopamine.configs import (
        InfiniDopamineTextConfig,
    )

    cfg = InfiniDopamineTextConfig()
    with pytest.raises(AttributeError, match="norm_topk_prob.*not supported"):
        # __get__ raises when accessed.
        _ = cfg.norm_topk_prob


def test_reading_after_set_still_raises() -> None:
    """The parent ``__init__`` may stash a default on the instance; the
    read path must still raise. This is the contract: any access to an
    unsupported field fails loudly regardless of how it got there.
    """
    from qwendopamine.models.infinidopamine.configs import (
        InfiniDopamineTextConfig,
    )

    cfg = InfiniDopamineTextConfig()
    # The parent's __init__ may have written to cfg.__dict__ directly
    # (bypassing the descriptor). Bypass the descriptor too.
    object.__setattr__(cfg, "norm_topk_prob", True)
    with pytest.raises(AttributeError, match="not supported"):
        _ = cfg.norm_topk_prob
