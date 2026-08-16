# QwenDopamine

QwenDopamine is a PyTorch research framework for Qwen-style LLM architectures, designed to evaluate novel recurrent memory blocks, altered state dynamics, and agentic performance enhancements.

It provides a pure PyTorch training and evaluation harness paired with Hugging Face `transformers` and GGUF weight interoperability.

## Key Features

- **Novel Recurrent Token Mixers**: Modular block architecture supporting pure-recurrent layers, including `GatedSurpriseNetAdam` (closed-form algebraic surprise fast-weights) and `GatedDeltaNet2` (`gdn2`).
- **Transformers & GGUF Interoperability**: Modular Qwen3.5 decoder alignment, native `safetensors` support, and dynamic GGUF weight tensor mapping.
- **CPU-First Local Environment**: Local development resolves against CPU PyTorch wheels via `uv`, isolating GPU dependencies (`flash-linear-attention`, `unsloth`) to remote/Kaggle environments to avoid Triton driver conflicts.
- **Production GPU Training Pipeline**: Multi-GPU DDP training scripts with 8-bit `PagedAdamW`, gradient checkpointing, automatic mixed precision (AMP), and early stopping.
- **Strict Quality Gates**: Complete behavioral test suite with `pyrefly` type checking and `ruff` linting.

## Repository Structure

- `annotations/` — experimental logs and run diagnostic annotations.
- `assets/` — metrics plots, visualization charts, and benchmark figures.
- `configs/` — Hydra configuration hierarchy for models, training protocols, and ablation studies.
- `docs/` — design specifications, architecture documentation, and run history insights.
- `notebooks/` — GPU verification scripts (e.g. `test_gated_surprise_net_gpu.py`, `test_gdn2_gpu.py`).
- `scripts/` — shell utility wrappers for training and evaluation.
- `src/qwendopamine/` — core Python package:
  - `models/` — model factory, core layers, modular Qwen3.5, and block registry.
  - `integrations/` — Hugging Face, GGUF, safetensors, and tokenizer loaders.
  - `training/` — training loop, learning rate schedules, freezing logic, and metrics.
  - `evaluation/` — perplexity computation, generation, and layerwise stats.
  - `distributed/` — process group setup and DDP utilities.
  - `cli/` — Hydra CLI entrypoints (`train.py`, `evaluate.py`, `convert.py`).
- `tests/` — behavioral pytest suite covering models, schedules, normalization, and weight loaders.

## Setup & Usage

### Local CPU Environment

```bash
# Sync local CPU environment with dev & HF dependencies
uv sync --extra cpu --extra dev --extra hf

# Run package CLI entrypoint
uv run qwendopamine

# Run Hydra training CLI
uv run src/qwendopamine/cli/train.py

# Run quality gates & tests
uv run pyrefly check
uv run ruff check .
uv run pytest -v
```

### GPU Pre-Training (2x T4 DDP / Single GPU)

```bash
# 2x T4 DDP training via torchrun
torchrun --nproc_per_node=2 notebooks/test_gated_surprise_net_gpu.py
```

## Development Conventions

- Use standard Hugging Face `transformers` for tokenization, generation, KV caching (`DynamicCache`), and checkpoint publishing.
- Use pure PyTorch for novel research blocks, altered residual connections, state dynamics, custom losses, and training loops.
- Keep `logs/`, `data/`, `checkpoints/`, `.venv/`, and `.aislop/` out of Git history.

