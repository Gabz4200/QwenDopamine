#!/usr/bin/env python3
"""Kaggle-mimic sandbox test: forces notebook's cell 1 install, then trains.

Emulates Kaggle by setting:
  KAGGLE_KERNEL_RUN=true
  QWD_CAPPED_FULL_PIPELINE=1
  QWD_LOCAL_STEPS=2
  QWD_SKIP_INSTALL=0 (forces cell 1 install step to run)
  QWD_PACKAGE_SPEC=<local repo>[gpu,cpt,hf]

Cell 1 uses `uv pip install --python sys.executable --break-system-packages --upgrade`,
which installs packages directly into the sandbox venv without touching system python.

In Kaggle batch/commit mode the notebook's setup cell calls os._exit(0) after
installing the numpy/scipy/sklearn trio to force a kernel restart. This script
emulates that by running the notebook in a child process: first run installs
and exits (simulating restart), second run picks up the fresh ABI-matched trio.

Usage:
  bash sandbox/test.sh
  sandbox/.venv-kaggle/bin/python sandbox/run.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

logger = __import__("logging").getLogger(__name__)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("KAGGLE_KERNEL_RUN", "true")
    os.environ.setdefault("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    os.environ.setdefault("QWD_CAPPED_FULL_PIPELINE", "1")
    os.environ.setdefault("QWD_CAPPED_ROWS", "5")
    os.environ.setdefault("QWD_LOCAL_STEPS", "2")
    os.environ.setdefault(
        "QWD_LOCAL_RUN_DIR", str((repo_root / "sandbox" / "runs").resolve())
    )
    os.environ.setdefault(
        "KAGGLE_WORKING_DIR", str(Path("/tmp/kaggle_sandbox").resolve())
    )
    os.environ.setdefault("QWD_DEBUG_INSTALL", "0")
    os.environ["QWD_SKIP_INSTALL"] = "0"
    os.environ.setdefault("QWD_PACKAGE_SPEC", f"{repo_root}[gpu,cpt,hf]")

    Path(os.environ["KAGGLE_WORKING_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["QWD_LOCAL_RUN_DIR"]).mkdir(parents=True, exist_ok=True)

    notebook = repo_root / "notebooks" / "train-infini-dopamine.py"
    assert notebook.exists(), f"Notebook not found at {notebook}"

    # First run: cell 1 installs trio + package, then os._exit(0) to simulate
    # a Kaggle kernel restart. We run it as a subprocess so the os._exit only
    # kills the child, not this harness.
    _env = dict(os.environ)
    _env["KAGGLE_KERNEL_RUN"] = "true"
    _env["QWD_SKIP_INSTALL"] = "0"

    print("[sandbox] Run 1: install + kernel restart simulation")
    _first = subprocess.run(
        [sys.executable, str(notebook)],
        env=_env,
        cwd=str(repo_root),
        capture_output=False,
        check=False,
    )

    print(f"[sandbox] first run exited with {_first.returncode}")
    # os._exit(0) in the notebook → code 0 (simulated restart).
    # A non-zero exit is a real failure.
    if _first.returncode != 0:
        sys.exit(f"[sandbox] first run (install) failed: {_first.returncode}")

    # Second run: kernel "restarted", trio healthy → install skipped.
    # Capture output so we can assert on it.
    print("[sandbox] Run 2: execution (trio fresh, install skipped)")

    _env2 = dict(os.environ)
    _env2["KAGGLE_KERNEL_RUN"] = "true"
    _env2["QWD_SKIP_INSTALL"] = "0"

    _second = subprocess.run(
        [sys.executable, str(notebook)],
        env=_env2,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    out = _second.stdout
    if _second.returncode != 0:
        print(out)
        print(_second.stderr[-2000:] if _second.stderr else "")
        sys.exit(
            f"[sandbox] second run (training) failed: {_second.returncode}"
        )

    print(out)
    from qwendopamine.testing.cpt_helpers import losses_from_trainer_state

    td_path = Path(os.environ["QWD_LOCAL_RUN_DIR"])
    runs = list(td_path.iterdir())
    assert runs, f"expected one run dir, got {runs}"
    run_dir = runs[0]
    print(f"[sandbox] run_dir: {run_dir}")

    losses = losses_from_trainer_state(run_dir)
    print(f"[sandbox] losses: {losses}")
    assert "Taichi arch" in out, "Taichi arch missing"
    assert "Taichi delta probe" in out, "delta probe missing"
    assert len(losses) >= 2, f"Expected at least 2 loss values, got {losses}"
    assert all(__import__("math").isfinite(l) for l in losses), (
        f"Non-finite loss: {losses}"
    )
    print("[sandbox] OK — Kaggle-mimic test (install + training) succeeded")


if __name__ == "__main__":
    main()
