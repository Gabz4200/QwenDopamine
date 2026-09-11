#!/usr/bin/env bash
set -euo pipefail
# Run the notebook in the sandbox, emulating Kaggle execution including dependency installation.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/sandbox/.venv-kaggle"

if [ ! -d "$VENV" ]; then
  echo "[sandbox] venv not found at $VENV, running setup.sh first..."
  bash "$ROOT/sandbox/setup.sh"
fi

echo "[sandbox] executing notebook test inside sandbox venv: $VENV"
"$VENV/bin/python" "$ROOT/sandbox/run.py"
