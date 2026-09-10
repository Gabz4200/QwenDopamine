#!/usr/bin/env bash
set -euo pipefail
# Minimal Kaggle-mimic venv: cuda (cu128) + cpt + hf + dev, isolated from main .venv (cpu).
# Re-uses uv cache, so second sync is fast. Requires uv and Python 3.12.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/sandbox/.venv-kaggle"

if [ ! -d "$VENV" ]; then
  echo "[sandbox] creating $VENV"
  uv venv "$VENV" --python 3.12 --seed
fi

echo "[sandbox] syncing Kaggle deps (cuda,cpt,hf,dev) into $VENV — ~2 GB, first run downloads cu128 wheels"
VIRTUAL_ENV="$VENV" uv sync --active --extra cuda --extra cpt --extra hf --extra dev

echo "[sandbox] done. Test with:"
echo "  $VENV/bin/python sandbox/run.py"
echo "  # or capped Kaggle-mimic via main venv:"
echo "  QWD_CAPPED_FULL_PIPELINE=1 KAGGLE_KERNEL_RUN=true uv run python sandbox/run.py"
