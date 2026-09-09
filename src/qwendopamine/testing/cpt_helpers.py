"""Helpers for CPT notebook integration tests.

Lives in src so pyrefly can resolve it (project import root is src).
Tests import from here directly; tests/conftest re-exports for convenience.
"""

from __future__ import annotations

import json
import logging
import math
import os
import runpy
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_cpt_notebook(
    tmp_path: Path,
    capsys,  # pytest CaptureFixture
    extra_env: dict[str, str] | None = None,
) -> tuple[str, Path]:
    """Run the CPT notebook as a script with isolated env.

    Returns (captured_stdout, run_dir).
    """
    notebook = Path("notebooks/train-infini-dopamine.py")
    assert notebook.exists(), "CPT notebook source missing"

    env = dict(os.environ)
    env["QWD_LOCAL_STEPS"] = "2"
    env["QWD_LOCAL_RUN_DIR"] = str(tmp_path)
    env.pop("KAGGLE_KERNEL_RUN", None)
    if extra_env:
        env.update(extra_env)

    old_argv, old_env = sys.argv, dict(os.environ)
    sys.argv = [str(notebook)]
    os.environ.clear()
    os.environ.update(env)
    try:
        runpy.run_path(str(notebook), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.environ.clear()
        os.environ.update(old_env)

    out = capsys.readouterr().out
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1, f"expected one run dir, got {runs}"
    return out, runs[0]


def losses_from_trainer_state(run_dir: Path) -> list[float]:
    """Parse losses from checkpoint trainer_state.json."""
    losses: list[float] = []
    for ckpt in sorted(run_dir.glob("checkpoint-*")):
        state_file = ckpt / "trainer_state.json"
        if not state_file.is_file():
            continue
        try:
            data = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping corrupt trainer_state %s: %s", state_file, exc)
            continue
        for entry in data.get("log_history", []):
            if "loss" in entry:
                try:
                    v = float(entry["loss"])
                except (ValueError, TypeError) as exc:
                    logger.debug("Skipping non-numeric loss %r: %s", entry.get("loss"), exc)
                    continue
                if math.isfinite(v):
                    losses.append(v)
                continue
            for k in ("train_loss", "eval_loss"):
                if k in entry:
                    try:
                        v = float(entry[k])
                    except (ValueError, TypeError) as exc:
                        logger.debug("Skipping non-numeric %s %r: %s", k, entry.get(k), exc)
                        continue
                    if math.isfinite(v):
                        losses.append(v)
    return losses
