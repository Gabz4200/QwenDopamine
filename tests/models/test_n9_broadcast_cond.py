"""N9: de-duplicate _broadcast_cond across reward block files."""

from __future__ import annotations


def test_components_module_exposes_broadcast_cond() -> None:
    """The shared ``broadcast_cond`` helper must be exposed from components."""
    from qwendopamine.models.blocks.reward import components

    assert hasattr(components, "broadcast_cond")
    assert callable(components.broadcast_cond)


def test_per_class_methods_no_longer_contain_inline_body() -> None:
    """The per-class _broadcast_cond wrappers are gone entirely (review N9).

    Only :class:`TokenWiseFiLM` (``film.py``) still calls the shared
    ``broadcast_cond`` helper; the other three classes never invoked it
    and only carried the dead wrapper.
    """
    import inspect

    from qwendopamine.models.blocks.reward import fourier, scalers
    from qwendopamine.models.blocks.reward.fourier import LearnableFourierFeatures
    from qwendopamine.models.blocks.reward.scalers import (
        AsinhScaler,
        LearnableSoftsign,
    )

    # No class may carry a _broadcast_cond method.
    for cls in (AsinhScaler, LearnableSoftsign, LearnableFourierFeatures):
        assert not hasattr(cls, "_broadcast_cond"), (
            f"{cls.__name__} still defines dead _broadcast_cond; "
            f"review N9: use the shared broadcast_cond helper."
        )
    # And no module may define the inline broadcast body anymore.
    for mod in (fourier, scalers):
        src = inspect.getsource(mod)
        assert "elif x.dim() == 2 and cond.dim() == 3" not in src, (
            f"{mod.__name__} still contains the inline _broadcast_cond body; "
            f"review N9: consolidate into broadcast_cond helper."
        )
