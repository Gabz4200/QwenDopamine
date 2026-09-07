"""N9: de-duplicate _broadcast_cond across reward block files."""

from __future__ import annotations


def test_components_module_exposes_broadcast_cond() -> None:
    """The shared ``broadcast_cond`` helper must be exposed from components."""
    from qwendopamine.models.blocks.reward import components

    assert hasattr(components, "broadcast_cond")
    assert callable(components.broadcast_cond)


def test_per_class_methods_no_longer_contain_inline_body() -> None:
    """The per-class _broadcast_cond methods must be thin wrappers, not
    contain the full inline body. Review N9: 4 copies were consolidated
    into one helper.
    """
    import inspect

    from qwendopamine.models.blocks.reward import film, fourier, scalers

    for mod in (film, fourier, scalers):
        src = inspect.getsource(mod)
        # Count occurrences of the inline "elif x.dim() == 2 and cond.dim() == 3"
        # in this module. We expect 0 (all replaced with thin wrappers).
        assert "elif x.dim() == 2 and cond.dim() == 3" not in src, (
            f"{mod.__name__} still contains the inline _broadcast_cond body; "
            f"review N9: consolidate into broadcast_cond helper."
        )
