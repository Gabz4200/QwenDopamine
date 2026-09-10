"""TDD for Kaggle notebook setup: ensure install is safe.

Seams under test:
- notebooks/train-infini-dopamine.py: Kaggle install command (file content)
- pyproject.toml: dependency version constraints
"""

from pathlib import Path

NOTEBOOK_PY = Path("notebooks/train-infini-dopamine.py")
PYPROJECT = Path("pyproject.toml")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_kaggle_install_does_not_force_full_transitive_upgrade() -> None:
    """Kaggle setup must not `uv pip install --upgrade --refresh --reinstall`.

    That triple forces uv to re-resolve and upgrade all 165+ transitive deps
    (numpy 2.5.3 + scipy 1.18.1) which breaks the `numpy._core.umath._center`
    private API via `peft -> transformers -> sklearn -> scipy -> numpy`.
    The fix is to install only the package without forcing transitive upgrades,
    e.g. `uv pip install --system qwendopamine[...] @ git+...` or limited
    `--refresh-package qwendopamine`.
    """
    text = _read(NOTEBOOK_PY)
    has_upgrade = "--upgrade" in text
    import re

    has_bare_refresh = bool(re.search(r'"--refresh"\s*,', text))
    has_reinstall = "--reinstall" in text
    bad = has_upgrade and has_bare_refresh and has_reinstall
    assert not bad, (
        "notebook still uses `uv pip install --upgrade --refresh --reinstall` "
        "which forces full transitive upgrade and breaks numpy/scipy compat "
        "(ImportError: cannot import name '_center' from 'numpy._core.umath'). "
        "Use a constrained install without bare --upgrade/--refresh/--reinstall."
    )


def test_kaggle_install_fallback_pip_not_forced_upgrade() -> None:
    """Fallback pip path also must not use `--upgrade` without constraints."""
    text = _read(NOTEBOOK_PY)
    import re

    has_pip_upgrade = bool(re.search(r'sys\.executable.*pip.*--upgrade', text, re.DOTALL))
    assert not has_pip_upgrade, (
        "pip fallback still uses --upgrade which forces transitive upgrade churn; "
        "remove --upgrade or constrain versions."
    )


def test_pyproject_pins_numpy_to_compatible_range() -> None:
    """pyproject must pin numpy to avoid broken 2.5.3 + scipy combo.

    Before fix: `numpy>=2.0.0` with no upper bound allowed uv to pull 2.5.3
    which breaks scipy's `numpy._core.umath._center` import.
    After fix: must have an upper bound like `<2.5` or `<2.6` or explicit `==`.
    """
    text = _read(PYPROJECT)
    import re

    m = re.search(r'"numpy([^"]*)"', text)
    assert m, "numpy dependency not found in pyproject"
    spec = m.group(1)
    assert "<" in spec or "==" in spec, (
        f"numpy spec {spec!r} has no upper bound; pin to avoid 2.5.3 breakage "
        "(e.g. 'numpy>=2.0.0,<2.5')"
    )


def test_pyproject_pins_scipy_to_compatible_range() -> None:
    """pyproject must pin scipy similarly."""
    text = _read(PYPROJECT)
    import re

    m = re.search(r'"scipy([^"]*)"', text)
    assert m, "scipy dependency not found in pyproject"
    spec = m.group(1)
    assert "<" in spec or "==" in spec, (
        f"scipy spec {spec!r} has no upper bound; pin to avoid 1.18 breakage "
        "(e.g. 'scipy>=1.13.0,<1.15' or at least <1.18)"
    )
