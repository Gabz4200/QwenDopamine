# AGENTS.md

This file gives the context a coding agent needs to work in the **QwenDopamine** repository.

## Project overview

`qwendopamine` is a PyTorch research framework for Qwen-style LLM architectures. It implements dual-stream Infini-attention, GDN-2 (arXiv:2605.22791), a Gated Reward Net, sliding window attention, Taichi-accelerated kernels, and Hugging Face / GGUF interop.

Key facts:

- Package: `qwendopamine`
- Default upstream base model: `Qwen/Qwen3.5-0.8B` (`qwendopamine.DEFAULT_QWEN35_REPO`)
- Python: `>=3.12,<3.14`
- Package manager: `uv` (lockfile `uv.lock` is committed)
- Torch is pulled via extras (`cpu` / `cu128` / `gpu`), never hard-pinned in core deps
- No `.github/`, no root `Makefile`, no in-tree CI. Local `uv` commands are the quality gate.
- A `src/Makefile` exists with convenience targets (`test`, `lint`, `typecheck`, `format`, install variants) — see [src/Makefile](src/Makefile).

### Setup

Use `uv`. CPU is the supported dev environment. Install core + dev + HF extras together:

```bash
uv sync --extra cpu --extra dev --extra hf
```

GPU extras are optional for normal code/test work:

```bash
uv sync --extra gpu --extra dev --extra hf
```

Multimodal CPT notebook extras:

```bash
uv sync --extra cpt --extra hf
```

Note: `cpu`, `cu128`, and `gpu` extras conflict (declared via `[tool.uv] conflicts`). Pick one torch variant per environment. The `pytorch-cpu` and `pytorch-cu128` indexes feed `cpu`/`cu128`/`gpu` torch wheels respectively.

### Entrypoints

```bash
uv run src/qwendopamine/cli/train.py
uv run qwendopamine
```

`import qwendopamine` is intentionally cheap: `qwendopamine.__init__` exposes `__version__` and `DEFAULT_QWEN35_REPO` and lazily dispatches `main()` to `qwendopamine.cli.train` so the full Hydra/transformers stack is not imported on bare import.

## Quality gates

Before committing, run:

```bash
uv run ruff check --fix .
uv run aislop scan
uv run pytest
uv run pyrefly check
```

- `uv run ruff check --fix .` — linter and formatter; default rules; `--fix` auto-fixes.
- `uv run aislop scan` — agentic quality gate; must be 0 errors (warnings are medium-confidence).
- `uv run pyrefly check` — type checker; `pyrefly.toml` is committed with ~113 suppressed categories tuned for third-party HF model classes and Taichi's dynamic runtime.
- `uv run pytest` — full suite is the verification target. The `slow` marker covers tests that download `Qwen/Qwen3.5-0.8B` weights; use `uv run pytest -m "not slow"` for quick local iteration.

`src/Makefile` also provides: `install`/`install-cpu`/`install-gpu`/`install-hf`/`install-cpt`/`install-dev`, `test`, `test-fast` (`-m "not slow"`), `test-slow`, `lint`, `typecheck`, `format`, `clean`.

## Layout

```
src/qwendopamine/
├── __init__.py            # __version__, DEFAULT_QWEN35_REPO, main() lazy dispatcher
├── cli/                   # Hydra entrypoint (train.py)
├── distributed/           # distributed setup
├── evaluation/            # perplexity, generation, layerwise stats
├── integrations/          # HF / GGUF / safetensors / tokenizer
│   ├── huggingface/       # HF model building, loading, registration, configs
│   ├── pytorch/           # torch autograd, chunking, custom ops, delta
│   ├── cpt_datasets.py    # streaming CPT dataset mixer (16 datasets)
│   ├── gguf.py            # GGUF export / import
│   ├── safetensors.py
│   └── tokenizer.py
├── kernels/taichi/        # Taichi kernels (forward + backward via autograd.Function)
├── models/
│   ├── blocks/            # BlockRegistry + build_block(...)
│   ├── core/              # embeddings, RMSNorm, LM head, config_adapter
│   ├── gdn2/              # GDN-2 blocks, recurrence (chunk + recurrent paths)
│   ├── gdn2_gpt/          # GDN-2 GPT variant
│   ├── infinidopamine/    # model, configs, HF weight translations
│   ├── qwen35/            # Qwen3.5 baseline
│   ├── reinforced/        # GatedRewardNet / Reinforced-Delta references
│   ├── shared/            # shared heads, pretrained mixins, text/vision towers
│   └── model_factory.py   # create_model / build_model
├── ops/                   # public operations layer (models import from here)
│   ├── gdn2.py
│   ├── reward.py
│   ├── _backend_registry.py
│   └── references/
├── testing/               # cpt_helpers (losses_from_trainer_state, run_cpt_notebook)
├── training/              # loop, schedules, freezing, parallel reward, metrics
└── utils.py
configs/                   # Hydra hierarchy (model / train / data / experiment)
tests/                     # pytest suite
notebooks/                 # multimodal CPT notebook (Jupytext-paired)
outputs/                   # Hydra run outputs (gitignored in practice)
annotations/               # design notes & bounds documentation
```

## Package-level rules

`src/qwendopamine/__init__.py`:
- exposes `__version__` and `DEFAULT_QWEN35_REPO`
- lazily dispatches `main()` to `qwendopamine.cli.train` to avoid importing the full Hydra/transformers stack on `import qwendopamine`

`src/qwendopamine/models/__init__.py`:
- uses PEP 562 lazy `__getattr__`
- importing `qwendopamine.models` is cheap; touching a model class loads its submodule on first access
- do NOT replace lazy imports with eager imports

### Dependency direction

```
models/* --> ops/* --> kernels/taichi/*
           |-> models/gdn2/recurrence/*  (pure-PyTorch reference)
           |-> models/reinforced/canonical_reference.py
```

Models MUST import operations from `qwendopamine.ops`, never directly from `qwendopamine.kernels.taichi.*`. This contract is documented in the ops package docstring.

### Registries

**`BlockRegistry`** (`src/qwendopamine/models/blocks/registry.py`): maps Hydra block names to layer implementations via `build_block(...)`. Lazy `_populate()` loads block classes on first access to avoid circular imports. Experiment YAML names MUST match registered block names. Layer-type selection is explicit via `config.layer_types[layer_idx]`; no implicit swap based on neighboring layers.

**`BackendRegistry`** (`src/qwendopamine/ops/_backend_registry.py`): replaces duplicated `_is_available()` branches. API: `register_backend(name, factory)`, `resolve_backend(name) -> str`, `available_backends() -> list[str]`, raises `BackendResolutionError`. `gdn2.py` and `reward.py` both dispatch through it with registered backends `torch-chunk`, `torch-recurrent`, `torch`, `taichi`, and `auto`.

**`ModelRegistry`** (`src/qwendopamine/models/model_factory.py`): `register_model_family(name, builder)` and `create_model(name, config, **kwargs)` for dynamic model loading (timm-style). Built-in families: `qwen35`, `infinidopamine`, `research`. `build_model(config)` delegates to `create_model` after resolving family from config.

## SOLID splits (no backward-compat shims)

Previously one large class, now composed of separate modules. `__init__.py` re-exports for package API convenience; there are no backward-compat shim files:

- `infinidopamine/` — `configs.py`, `model_impl.py`, `model_outputs.py`, `decoder_layer.py`, `_gated_delta_net.py`, `_gated_reward_net.py`, `_attention.py`, `_mlp.py`, `_norm.py`, `rotary_embeddings.py`
- `shared/` — `heads_causal_lm.py`, `heads_token_classification.py`, `heads_conditional_generation.py`, `heads_sequence_classification.py`, `model.py`, `outputs.py`, `pretrained.py`, `text.py`, `vision.py`
- `gdn2_gpt/` — `params.py` (`compute_model_params`); `model.py` (`GDN2GPT`)
- `reinforced/` — `_normalizer.py` extracted from `_layer.py`
- `gdn2/` — `_block_meta.py` holds default constants for SRP

Import directly from the specific module, or from the package `__init__.py` for the public surface.

## Testing

Confirm the resolved Taichi backend when needed:

```bash
uv run python -c "from qwendopamine.kernels.taichi import taichi_arch; print(taichi_arch())"
```

Expected returns: `cpu`, `cuda`, `vulkan`, `metal`, `opengl`, or `unavailable` (when Taichi cannot initialise). On this machine it resolves to `vulkan`.

Focused runs:

```bash
uv run pytest -v
uv run pytest -m "not slow" -v

uv run pytest tests/models/test_gdn2.py -v
uv run pytest tests/models/test_taichi_gdn2.py tests/models/test_taichi_gdn2_backward.py -v
uv run pytest tests/models/test_reward*.py -v
uv run pytest tests/models/test_reinforced_delta_layer_taichi_path.py -v
uv run pytest tests/ops/ -v
uv run pytest tests/kernels/ -v
```

Pytest is configured in `pyproject.toml` (`testpaths = ["tests"]`, `slow` marker declared). The `testing` submodule exposes `losses_from_trainer_state` and `run_cpt_notebook`.

## Taichi invariants

- `src/qwendopamine/kernels/taichi/runtime.py` is the ONLY place that calls `ti.init()`. It tries `ti.init(arch=ti.gpu, default_fp=ti.f32)` (letting Taichi pick CUDA → Vulkan → Metal/OpenGL → CPU) and falls back to `ti.init(default_fp=ti.f32)` if that fails. The resolved arch is cached in `_ARCH` and exposed via `taichi_arch()`.
- default float precision is `ti.f32`; do not switch a single kernel to `ti.f64`
- per-token replay is the only safe chunkwise adjoint pattern; do not hand-write a WY-bwd kernel
- the effective per-channel gate helper `_make_effective_gate(omega, gate) -> [B, D]` is the single contraction point for `omega * write`

## Core implementation rules

Reward branch:
- the parallel reward branch never replaces the main mixer
- it attaches only when listed in `config.parallel_reward_layers`, or when `use_parallel_reward=True` and the layer is attention-only

Reward state persistence:
- `GatedRewardNet.forward` returns recurrent/value/conv state
- the wrapper writes them into `DynamicCache` under reward-specific keys
- never clobber the GDN-2 cache

GDN-2 gate shape:

```python
# correct
einsum("bd,bdk,bk->bk", omega_W, dS, e)

# wrong
omega_W * einsum("bdk,bd->bk", dS, e)
```

Chain rule:
- given `omega_w_eff = omega_w * write`, recover `d_write = d_omega_w_eff * omega_w`
- do NOT divide by `write.clamp_min(1e-12)`

Low-precision numerics:
- upcast to `float32` before `.pow()`, `.sum()`, `.mean()` on `float16` / `bfloat16`

Optimizer/scheduler/global-step guard:
- skipped steps must not advance optimizer state, scheduler counters, or global step
- schedulers must serialize `step_count`, `phase`, and `warmup_steps`

HF config hierarchy:
- multimodal configs nest `text_config` and `vision_config`
- never read top-level `hidden_size` on `InfiniDopamineConfig`; always go through `text_config`

Qwen3.5 weight loading:
- loading uses `strict=True`
- new parameters need translation rules in `_qwen35_weights.py` / `_text_qwen35_weights.py`

## Workflows

### Add a new block

1. Implement under `src/qwendopamine/models/blocks/` or `src/qwendopamine/models/reinforced/`
2. Register in `src/qwendopamine/models/blocks/registry.py`
3. Add a Hydra preset under `configs/model/`
4. Add tests under `tests/models/`: forward shape, gradient flow, and if Taichi, forward/backward parity against the PyTorch reference
5. Update `README.md` if the architecture contract changes

### Add or change a Taichi kernel

1. Implement under `src/qwendopamine/kernels/taichi/`
2. Wrap forward/backward in a `torch.autograd.Function`
3. Expose through `src/qwendopamine/ops/`
4. Add parity tests in `tests/models/test_taichi_*.py`
5. Run focused Taichi tests plus `tests/ops/` and `tests/kernels/`

### Add a new Hugging Face weight translation

1. Edit `src/qwendopamine/models/infinidopamine/_qwen35_weights.py` and/or `_text_qwen35_weights.py`
2. Add a synthetic roundtrip test in `tests/models/test_hf_roundtrip_synthetic.py`
3. Add a slow `0.8B` load test if practical

### Add a training/eval feature

1. Implement under `src/qwendopamine/training/` or `src/qwendopamine/evaluation/`
2. Wire into `loop.py`
3. Ensure checkpoint/schedule state round-trips through `schedules.py`
4. Add Hydra config under `configs/train/` or `configs/experiment/`
5. Add tests in `tests/models/test_training.py` / `test_schedules.py`

### Notebook

`notebooks/train-infini-dopamine.ipynb` is Jupytext-paired with `notebooks/train-infini-dopamine.py`. It streams 16 HF datasets with per-dataset schema formatters and pushes merged checkpoints to the Hub via `accelerate.PartialState()`.

- Requires `[cpt]` extras: `uv sync --extra cpt --extra hf`
- On Kaggle, use `accelerate.PartialState()`, NOT `accelerate launch` / `torchrun`
- Use `git+https` or archive zip URLs; bare `.git` URLs break pip
- Keep the `.py` / `.ipynb` pair in sync with `jupytext --sync`
- Do not run unless explicitly asked (downloads large datasets, pushes checkpoints)

## Config hierarchy

Hydra configs live under `configs/` and are composed on the CLI:

| Directory | Presets |
|---|---|
| `configs/model/` | `base.yaml`, `qwen35_reference`, `qwen35_gdn2`, `qwen35_custom_block`, `infinidopamine_reference`, `infinidopamine_gdn2`, `infinidopamine_custom_block` |
| `configs/train/` | `cpu.yaml`, `single_gpu.yaml`, `fsdp2.yaml`, `frozen_backbone.yaml`, `cpu_frozen_backbone.yaml` |
| `configs/data/` | `pretrain.yaml`, `debug.yaml` |
| `configs/experiment/` | `ablation_block_position.yaml`, `ablation_unfreezing.yaml` |

```bash
# Compose presets on the command line (default config_path is configs/, config_name train/cpu)
uv run src/qwendopamine/cli/train.py model=qwen35_gdn2 train=cpu data=debug
```

## Boundaries and conventions

- never placeholders, em-dashes, or silent scope expansion
- never revert or modify code you did not write unless explicitly asked
- ask before destructive commands, dependency upgrades, or scope expansion
- fix root causes; do not catch `KeyError`, `IndexError`, `TypeError`, `AttributeError` for developer bugs (fail-fast; see `/home/gabz/.agents/rules/fail-fast.md`)
- test behavior at public boundaries, not internal implementation
- no `.github/`, no root `Makefile`, no in-tree CI — local `uv` commands are the quality gate

## Quality gate summary

| Tool | Command | Failure mode |
|---|---|---|
| ruff | `uv run ruff check --fix .` | lint/type errors |
| aislop | `uv run aislop scan` | must be 0 errors; warnings are medium-confidence |
| pyrefly | `uv run pyrefly check` | real type errors on public surface (third-party model inference noise suppressed in `pyrefly.toml`) |
| pytest | `uv run pytest` | full suite; `-m "not slow"` for local iteration |
