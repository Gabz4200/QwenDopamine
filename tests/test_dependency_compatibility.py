"""Tests for runtime dependency compatibility.

These tests verify that the pinned/installed dependency versions work
together, specifically guarding against the NumPy 2.x + SciPy <1.13
import failure mode observed on Kaggle:

    ImportError: cannot import name '_center' from 'numpy._core.umath'

The notebook install path and pyproject.toml must stay consistent; if
either drifts, these tests will fail.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_numpy_scipy_sklearn_import_chain() -> None:
    """The exact import chain that failed on Kaggle must succeed.

    scikit-learn is not a direct project dependency, so this test is
    skipped when it is not installed. The underlying guard is the
    ``numpy + scipy`` compatibility tests below.
    """
    importlib.import_module("numpy")
    importlib.import_module("scipy.sparse")
    pytest.importorskip("sklearn")
    importlib.import_module("sklearn.metrics")


def test_numpy_major_version_is_2() -> None:
    """NumPy 2.x is required; NumPy 1.x is out of support."""
    major = int(np.__version__.split(".")[0])
    assert major >= 2, f"NumPy 2.x required, got {np.__version__}"


def test_scipy_minimum_version() -> None:
    """SciPy must be new enough to support NumPy 2.x.

    SciPy < 1.13 lacks the NumPy 2.x compatibility shims and triggers
    the ``_center`` import error through ``scipy._lib.array_api_compat``.
    """
    import scipy
    from packaging.version import Version

    assert Version(scipy.__version__) >= Version("1.13.0"), (
        f"SciPy >=1.13.0 required for NumPy 2.x compatibility, got {scipy.__version__}"
    )


def test_notebook_pin_matches_project_pin() -> None:
    """The notebook install cell must stay consistent with pyproject pins.

    After the uv migration the notebook no longer lists numpy/scipy
    explicitly — it installs ``qwendopamine[cuda,cpt,hf] @ git+...`` and relies
    on pyproject's pins. This test guards against drift: if the notebook
    does list explicit pins they must match, otherwise it must use the
    package spec and pyproject must have the floor pins.
    """
    notebook_path = "notebooks/train-infini-dopamine.py"
    with open(notebook_path) as fh:
        source = fh.read()

    has_package_spec = (
        "qwendopamine[cuda,cpt,hf]" in source or "qwendopamine[" in source
    )
    numpy_line: str | None = None
    scipy_line: str | None = None
    for line in source.splitlines():
        stripped = line.strip().strip('",').strip("'")
        # Only consider dependency pin lines, not error messages
        if stripped.startswith("numpy") and ">=" in stripped:
            numpy_line = stripped
        elif stripped.startswith("scipy") and ">=" in stripped:
            scipy_line = stripped

    if numpy_line is not None:
        assert ">=2.0" in numpy_line, (
            f"Notebook numpy pin must be >=2.0 for NumPy 2.x compatibility, "
            f"found: {numpy_line!r}"
        )
    if scipy_line is not None:
        assert ">=1.15" in scipy_line or ">=1.13" in scipy_line, (
            f"Notebook scipy pin must be >=1.15 for NumPy 2.x compatibility, "
            f"found: {scipy_line!r}"
        )
    if numpy_line is None and scipy_line is None:
        assert has_package_spec, "notebook must install via qwendopamine package spec"
        # Verify pyproject still has the floor pins
        import tomllib

        with open("pyproject.toml", "rb") as pf:
            data = tomllib.load(pf)
        deps = " ".join(data.get("project", {}).get("dependencies", []))
        assert "numpy>=2.0.0" in deps
        assert "scipy>=1.15" in deps or "scipy>=1.13" in deps


def test_pyproject_toml_has_compatible_pins() -> None:
    """pyproject.toml must declare compatible numpy/scipy minimums.

    Review N11: previously the test asserted raw text (``"numpy>=2.0.0"
    in content``), which is brittle and breaks on formatting changes.
    The fix parses the file with ``tomllib`` and checks the parsed
    dependency strings.
    """
    import tomllib

    with open("pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)

    deps = data.get("project", {}).get("dependencies", [])
    dep_strs = " ".join(deps)
    assert "numpy>=2.0.0" in dep_strs, (
        f"pyproject.toml must require numpy>=2.0.0; got deps: {deps!r}"
    )
    assert "scipy>=1.15" in dep_strs, (
        f"pyproject.toml must require scipy>=1.15 for NumPy 2.x ABI; got deps: {deps!r}"
    )
