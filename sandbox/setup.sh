#!/usr/bin/env bash
set -euo pipefail
# Minimal sandbox: creates a clean, isolated venv for Kaggle emulation.
# The dependencies are installed by the notebook itself during the test run.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/sandbox/.venv-kaggle"

echo "[sandbox] creating clean venv at $VENV"
rm -rf "$VENV"
uv venv "$VENV" --python 3.12 --seed
# Seed venv is intentionally minimal (no packaging). Install bootstrap deps
# so notebook cell 1 can import Version before it installs the full stack.
"$VENV/bin/pip" install -q packaging
echo "[sandbox] clean venv ready at $VENV"
echo "[sandbox] run 'bash sandbox/test.sh' to test notebook dependency installation and execution"
