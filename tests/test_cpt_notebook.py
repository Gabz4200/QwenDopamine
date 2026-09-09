"""Training-loop smoke test for the CPT notebook.

Runs the exact notebook pipeline (same ``.py`` source Jupytext syncs to the
``.ipynb``) end to end on CPU: synthetic dataset, tiny random-init text
model, Taichi kernels mandatory, 2 optimizer steps. Marked slow because it
compiles Taichi kernels and runs a real trainer loop (~2 min).

The test asserts the training actually used Taichi: it checks the Taichi
arch line plus the delta probe ran in captured stdout, and requires a real
loss logged via trainer_state.json.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qwendopamine.testing.cpt_helpers import losses_from_trainer_state, run_cpt_notebook


@pytest.mark.slow
def test_when_cpt_notebook_then_trains_two_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, run_dir = run_cpt_notebook(tmp_path, capsys)

    assert "Taichi arch" in out, "notebook Taichi arch line did not print"
    assert "Taichi delta probe" in out, "reward Taichi probe did not run"

    losses = losses_from_trainer_state(run_dir)
    assert len(losses) >= 2, f"expected >=2 logged losses, got {losses}"
    assert all(math.isfinite(loss) for loss in losses), f"non-finite loss: {losses}"

    assert (run_dir / "checkpoint-2").is_dir(), f"no checkpoint-2 in {run_dir}"
    assert (run_dir / "peft-final").is_dir(), f"no peft-final in {run_dir}"
