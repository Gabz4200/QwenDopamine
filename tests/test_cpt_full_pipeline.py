"""Capped full-pipeline smoke test for the CPT notebook.

Exercises the full Kaggle dataset pipeline locally without network:
17 mocked streams (5 rows each) go through the real formatters,
interleave, tokenization, and a 2-step Taichi training loop.
Marked slow because it still compiles Taichi kernels (~20s).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qwendopamine.testing.cpt_helpers import losses_from_trainer_state, run_cpt_notebook


@pytest.mark.slow
def test_when_capped_full_pipeline_then_all_formatters_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, run_dir = run_cpt_notebook(
        tmp_path, capsys, extra_env={"QWD_CAPPED_FULL_PIPELINE": "1"}
    )

    assert "[capped-full]" in out, "capped-full pipeline marker missing"
    assert "Taichi arch" in out, "Taichi arch line missing"
    assert "Taichi delta probe" in out, "reward Taichi probe missing"

    losses = losses_from_trainer_state(run_dir)
    assert len(losses) >= 2, f"expected >=2 logged losses, got {losses}"
    assert all(math.isfinite(loss) for loss in losses), f"non-finite loss: {losses}"

    assert (run_dir / "checkpoint-2").is_dir(), f"no checkpoint-2 in {run_dir}"
    assert (run_dir / "peft-final").is_dir(), f"no peft-final in {run_dir}"
