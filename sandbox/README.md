# Sandbox — Kaggle-mimic local test

Minimal sandbox to test `notebooks/train-infini-dopamine.py` with the same deps Kaggle uses, without needing a GPU or real datasets.

## What it does

- Installs the Kaggle variant `qwendopamine[cuda,cpt,hf]` (cu128 torch) into `sandbox/.venv-kaggle` via `uv`, isolated from the main `cpu` dev env (`.venv`).
- Runs the notebook as a script with `KAGGLE_KERNEL_RUN=true` + `QWD_CAPPED_FULL_PIPELINE=1` (5 mocked rows per dataset, 17-way interleave) and 2 optimizer steps, tiny random-init model, `adamw_torch` on CPU. This exercises the Kaggle code path (install gated, `IS_KAGGLE` true, `TORCH_DTYPE` logic) but stays inside the local `runs/` footprint.

## Quick start

```bash
# 1. Create the Kaggle-deps venv (once, ~2 GB, re-uses uv cache)
bash sandbox/setup.sh

# 2. Run the capped Kaggle-mimic smoke (CPU, ~30s, no network after Pillow)
sandbox/.venv-kaggle/bin/python sandbox/run.py
# or without the separate venv, using the main cpu venv with mocks:
QWD_CAPPED_FULL_PIPELINE=1 QWD_CAPPED_ROWS=5 KAGGLE_KERNEL_RUN=true uv run python sandbox/run.py
```

## Flags

- `QWD_DEBUG_INSTALL=1` — force the `uv pip install qwendopamine[cuda,cpt,hf]` path even when `IS_KAGGLE` is false. Use in the sandbox to test the install logic without `/kaggle/working`.
- `QWD_CAPPED_FULL_PIPELINE=1` — use 5 mocked rows per dataset instead of streaming 17 real Hub datasets. Required for sandbox; otherwise the notebook streams the full 17.
- `QWD_CAPPED_ROWS=5` — rows per mocked dataset (default 5).
- `QWD_LOCAL_STEPS=2` — optimizer steps when `_USE_SMOKE_CONFIG` is true (Kaggle+capped or local).

## Layout divergence

After the refactor, Kaggle vs local is only one-liners on `IS_KAGGLE` / `_USE_SMOKE_CONFIG` (`not IS_KAGGLE or _CAPPED_FULL`):

```
IS_KAGGLE = os.path.isdir("/kaggle/working") or KAGGLE_KERNEL_RUN=="true"
_USE_SMOKE_CONFIG = not IS_KAGGLE or _CAPPED_FULL
TORCH_DTYPE = float32 if not cuda else (bf16 if bf16 else fp16)
MAX_TRAIN_STEPS = local_steps if _USE_SMOKE else None
... each batch/warmup/save/refresh flag is `x if _USE_SMOKE else y`
_SHOULD_INSTALL = IS_KAGGLE or QWD_DEBUG_INSTALL
```

`_RUN_ROOT` and the model/optim branches also use `_USE_SMOKE_CONFIG`, so `KAGGLE_KERNEL_RUN=true QWD_CAPPED_FULL_PIPELINE=1` locally behaves like Kaggle but with smoke footprint.

## Verification

```bash
uv run ruff check --fix . && uv run pyrefly check && uv run pytest -m "not slow" -q
uv run pytest tests/test_cpt_notebook.py -v  # 2-step local already
QWD_CAPPED_FULL_PIPELINE=1 KAGGLE_KERNEL_RUN=true uv run python sandbox/run.py  # Kaggle-mimic
```

The sandbox run prints `Taichi arch`, `Reward-values forward pass OK`, `Taichi delta probe`, and creates `runs/.../checkpoint-2` + `peft-final`.
