# AGENTS.md

Instructions for agents working on the **QwenDopamine** repository.

## Project overview

`qwendopamine` is a PyTorch research framework for Qwen-style LLM architectures with dual-stream Infini-attention, Gated Reward Net, GDN-2, sliding window attention, and Hugging Face/GGUF interop.

Key facts:

- Package: `qwendopamine`
- Default upstream base model: `Qwen/Qwen3.5-0.8B` (`qwendopamine.DEFAULT_QWEN35_REPO`)
- Python: `>=3.12,<3.14`
- Package manager: `uv`
- Lockfile: `uv.lock`
- No `.github/`, no `Makefile`, no in-tree CI. Local `uv` commands are the quality gate.

## Setup

Use `uv`. CPU is the supported dev environment.

```bash
uv sync --extra cpu --extra dev --extra hf
```

GPU extras are optional for normal code/test work:

```bash
uv sync --extra gpu
```

Entrypoints:

```bash
uv run src/qwendopamine/cli/train.py
uv run qwendopamine
```

## Quality gates

Run before committing (exact verification commands):

```bash
uv run ruff check --fix .
uv run aislop scan
uv run pytest
uv run pyrefly check
```

- `ruff` is the linter/formatter with default rules (`--fix` auto-fixes).
- `aislop` is the quality gate for agentic coding (must be 0 errors, warnings are medium-confidence).
- `pyrefly` is the type checker; `pyrefly.toml` is committed.
- Tests use `pytest`; the `slow` marker exists but full `uv run pytest` is the verification target (use `-m "not slow"` for quick focused runs).

## Layout

- `src/qwendopamine/models/...` - model code
- `src/qwendopamine/ops/` - public operations layer; models MUST import from here
- `src/qwendopamine/kernels/taichi/` - Taichi kernels
- `src/qwendopamine/training/` and `src/qwendopamine/evaluation/`
- `src/qwendopamine/cli/` - CLI entrypoints
- `src/qwendopamine/distributed/` - distributed setup
- `src/qwendopamine/integrations/` - HF/GGUF/safetensors/tokenizer interop
- `configs/` - Hydra configs
- `tests/` - pytest suite
- `notebooks/` - Jupytext-paired CPT notebook; requires `[cpt]`

## Package-level rules

`src/qwendopamine/__init__.py`:

- exposes `__version__` and `DEFAULT_QWEN35_REPO`
- lazily dispatches `main()` to `qwendopamine.cli.train` to avoid importing the full Hydra/transformers stack on `import qwendopamine`

`src/qwendopamine/models/__init__.py`:

- uses PEP 562 lazy `__getattr__`
- importing `qwendopamine.models` is cheap; touching a model class loads its submodule on first access
- do NOT replace lazy imports with eager imports

Dependency direction:

```
models/* --> ops/* --> kernels/taichi/*
           |-> models/gdn2/recurrence/*
           |-> models/reinforced/canonical_reference.py
```

Models MUST import operations from `qwendopamine.ops`, never directly from `qwendopamine.kernels.taichi.*`.

## Block registry

`src/qwendopamine/models/blocks/registry.py` defines `BlockRegistry` and `build_block(...)`.

- experiment YAML names MUST match registered block names
- unregistered names break Hydra instantiation
- layer-type selection is explicit via `config.layer_types[layer_idx]`; no implicit swap based on neighboring layers
- registry lazily populates on first access to avoid circular imports

## Dynamic registries

- `BlockRegistry` in `src/qwendopamine/models/blocks/registry.py`: lazy `_populate()` loads block classes on first access
- `BackendRegistry` in `src/qwendopamine/ops/_backend_registry.py`: replaces duplicated `_is_available()` branches; `gdn2.py` and `reward.py` dispatch through it
- `ModelRegistry` in `src/qwendopamine/models/model_factory.py`: `register_model_family(name, builder)` and `create_model(name, config)` for dynamic model loading (timm-style)

## Model factory

`src/qwendopamine/models/model_factory.py` is the composition root for dynamic model creation:

- `create_model(name, config, **kwargs)` instantiates a registered model family
- Built-in families: `qwen35`, `infinidopamine`, `research`
- `build_model(config)` delegates to `create_model` after resolving family from config
- Lazy imports keep `import qwendopamine.models` cheap

## SOLID splits (no backward-compat shims)

God classes were split into composable modules. `__init__.py` files re-export for package API convenience, but there are no backward-compat shim files:

- `infinidopamine/`: `configs.py`, `model_impl.py`, `model_outputs.py`, `decoder_layer.py`, `_gated_delta_net.py`, `_gated_reward_net.py`, `_attention.py`, `_mlp.py`, `_norm.py`, `rotary_embeddings.py`
- `shared/`: `heads_causal_lm.py`, `heads_token_classification.py`, `heads_conditional_generation.py`, `heads_sequence_classification.py`, `model.py`, `outputs.py`, `pretrained.py`, `text.py`, `vision.py`
- `gdn2_gpt/`: `params.py` holds `compute_model_params`; `model.py` keeps `GDN2GPT`
- `reinforced/`: `_normalizer.py` extracted from `_layer.py`
- `gdn2/`: `_block_meta.py` holds default constants for SRP

Import directly from the specific module, or from the package `__init__.py` for the public surface.

## Testing

Common focused runs:

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

Confirm the active Taichi backend when needed:

```bash
uv run python -c "from qwendopamine.kernels.taichi import taichi_arch; print(taichi_arch())"
```

## Taichi invariants

- `src/qwendopamine/kernels/taichi/runtime.py` is the ONLY place that calls `ti.init()`
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

`notebooks/train-infini-dopamine.ipynb` is Jupytext-paired with `notebooks/train-infini-dopamine.py`. It downloads datasets and pushes checkpoints to the Hub. Do not run unless explicitly asked.

## Boundaries

- never placeholders, em-dashes, or silent scope expansion
- never revert or modify code you did not write unless explicitly asked
- ask before destructive commands, dependency upgrades, or scope expansion
- fix root causes; do not catch `KeyError`, `IndexError`, `TypeError`, `AttributeError` for developer bugs
- test behavior at public boundaries, not internal implementation
