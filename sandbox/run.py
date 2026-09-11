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

Usage:
  bash sandbox/test.sh
  sandbox/.venv-kaggle/bin/python sandbox/run.py
"""

from __future__ import annotations

import importlib
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("KAGGLE_KERNEL_RUN", "true")
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

    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        env = dict(os.environ)
        env["QWD_LOCAL_STEPS"] = "2"
        env["QWD_LOCAL_RUN_DIR"] = str(td_path)
        env["QWD_CAPPED_FULL_PIPELINE"] = "1"
        env["QWD_CAPPED_ROWS"] = "5"
        env["KAGGLE_KERNEL_RUN"] = "true"
        env["KAGGLE_WORKING_DIR"] = str(Path("/tmp/kaggle_sandbox").resolve())
        env["QWD_SKIP_INSTALL"] = "0"
        env["QWD_PACKAGE_SPEC"] = f"{repo_root}[gpu,cpt,hf]"
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(k, None)

        try:
            from accelerate.state import PartialState

            PartialState._reset_state()
        except Exception as exc:  # noqa: BLE001
            logger.debug("PartialState reset skipped: %s", exc)
        try:
            import torch.distributed

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:  # noqa: BLE001
            logger.debug("destroy_process_group skipped: %s", exc)

        old_argv, old_env = sys.argv, dict(os.environ)
        sys.argv = [str(notebook)]
        os.environ.clear()
        os.environ.update(env)
        try:
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                _runpy = importlib.import_module("runpy")
                _runpy.run_path(str(notebook), run_name="__main__")
            finally:
                sys.stdout = old_stdout
            out = buf.getvalue()
        finally:
            sys.argv = old_argv
            os.environ.clear()
            os.environ.update(old_env)
            try:
                from accelerate.state import PartialState

                PartialState._reset_state()
            except Exception as exc:  # noqa: BLE001
                logger.debug("PartialState cleanup skipped: %s", exc)

        print(out)
        runs = list(td_path.iterdir())
        assert runs, f"expected one run dir, got {runs}"
        run_dir = runs[0]
        print(f"[sandbox] run_dir: {run_dir}")
        from qwendopamine.testing.cpt_helpers import losses_from_trainer_state

        losses = losses_from_trainer_state(run_dir)
        print(f"[sandbox] losses: {losses}")
        assert "Taichi arch" in out, "Taichi arch missing"
        assert "Taichi delta probe" in out, "delta probe missing"
        assert len(losses) >= 2, f"Expected at least 2 loss values, got {losses}"
        assert all(__import__("math").isfinite(l) for l in losses), (
            f"Non-finite loss: {losses}"
        )
        print("[sandbox] OK — Kaggle-mimic test (install + training) succeeded")
        dest = repo_root / "sandbox" / "runs" / run_dir.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        import shutil

        shutil.copytree(run_dir, dest)
        print(f"[sandbox] run results copied to {dest}")


if __name__ == "__main__":
    main()
