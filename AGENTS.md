# AGENTS.md

Instructions for agents working on the **QwenDopamine** repository.

## What this project is

A PyTorch research framework that re-implements Qwen-style LLM architectures
with dual-stream Infini-attention, Gated Reward Net, GDN-2, and sliding window
attention. It includes a pure-PyTorch training/eval harness with Hugging Face
`transformers` and GGUF weight interop.

- Package: `qwendopamine`
- Default upstream weights: `Qwen/Qwen3.5-0.8B` (`qwendopamine.DEFAULT_QWEN35_REPO`)
- Python: `>=3.12,<3.14`
- Lockfile: `uv.lock`
- No `.github/`, no `Makefile`, no in-tree CI. Local `uv` commands are the quality gate.

## Setup

Use `uv`. The supported dev environment is CPU-only.

```bash
uv sync --extra cpu --extra dev --extra hf
```

GPU extras are available but optional for normal code/test work:

```bash
uv sync --extra gpu
```

Run the entrypoints:

```bash
uv run src/qwendopamine/cli/train.py
uv run qwendopamine
```

## Quality gates

Run before committing:

```bash
uv run ruff check .
uv run pyrefly check
uv run pytest -m "not slow" -v
```

`ruff` is the formatter/linter, with defaults. `pyrefly` is the type checker;
`pyrefly.toml` is committed.

## Layout

- `src/qwendopamine/models/...` - model code
- `src/qwendopamine/ops/` - public operations layer; models import from here
- `src/qwendopamine/kernels/taichi/` - Taichi kernels
- `src/qwendopamine/training/` and `src/qwendopamine/evaluation/`
- `configs/` - Hydra configs
- `tests/` - pytest suite
- `notebooks/` - multimodal CPT notebook; requires `[cpt]`

### Lazy imports

`qwendopamine.models.__init__` uses **PEP 562 lazy `__getattr__`**.
`import qwendopamine.models` is ~0.0s; accessing a model class loads its
submodule on first access. Do not replace this with eager imports.

## Testing

Pytest is configured in `pyproject.toml`. Marker `slow` skips tests that
download Qwen3.5-0.8B weights.

```bash
uv run pytest -v
uv run pytest -m "not slow" -v
```

Common focuses:

```bash
uv run pytest tests/models/test_gdn2.py -v
uv run pytest tests/models/test_taichi_gdn2.py tests/models/test_taichi_gdn2_backward.py -v
uv run pytest tests/models/test_reward*.py -v
uv run pytest tests/models/test_reinforced_delta_layer_taichi_path.py -v
uv run pytest tests/ops/ -v
uv run pytest tests/kernels/ -v
```

Confirm the Taichi backend when needed:

```bash
uv run python -c "from qwendopamine.kernels.taichi import taichi_arch; print(taichi_arch())"
```

## Taichi invariants

- `kernels/taichi/runtime.py` is the **only** place that calls `ti.init()`.
- Models MUST import from `qwendopamine.ops`, never from `qwendopamine.kernels.taichi.*`.
- Per-token replay is the only safe chunkwise adjoint pattern. Do not hand-write a WY-bwd kernel.
- The effective per-channel gate helper `_make_effective_gate(omega, gate) -> [B, D]`
  is the single contraction point for `omega * write`.
- Default fp is `ti.f32`. Do not switch a single kernel to `ti.f64`.

## Core rules

Block selection: layer-type selection is explicit via `config.layer_types[layer_idx]`.
No block is swapped implicitly based on neighboring layers.

Reward branch: the parallel reward branch never replaces the main mixer. It
attaches only when listed in `config.parallel_reward_layers`, or when
`use_parallel_reward=True` and the layer is attention-only.

Reward state persistence: `GatedRewardNet.forward` returns recurrent/value/conv
state. The wrapper writes them into `DynamicCache` under reward-specific keys.
Never clobber the GDN-2 cache.

GDN-2 gate shape: per-channel weights live inside the einsum:

```python
# correct
einsum("bd,bdk,bk->bk", omega_W, dS, e)

# wrong
omega_W * einsum("bdk,bd->bk", dS, e)
```

Chain rule: given `omega_w_eff = omega_w * write`, recover
`d_write = d_omega_w_eff * omega_w`. Do not divide by
`write.clamp_min(1e-12)`.

Low-precision numerics: upcast to `float32` before `.pow()`, `.sum()`,
`.mean()` on `float16` / `bfloat16`.

Optimizer/scheduler/global-step guard: skipped steps must not advance optimizer
state, scheduler counters, or global step. Schedulers must serialize
`step_count`, `phase`, and `warmup_steps`.

HF config hierarchy: multimodal configs nest `text_config` and `vision_config`.
Never read top-level `hidden_size` on `InfiniDopamineConfig`; go through
`text_config`.

Experiment YAMLs: names must match registered blocks in
`src/qwendopamine/models/blocks/registry.py` (`BLOCKS`). Unregistered names
break Hydra instantiation.

Qwen3.5 weight loading: loading uses `strict=True`. New parameters need
translation rules in `_qwen35_weights.py` / `_text_qwen35_weights.py`.

## Workflows

Add a new block:

1. Implement under `src/qwendopamine/models/blocks/` or `models/reinforced/`
2. Register in `src/qwendopamine/models/blocks/registry.py`
3. Add a Hydra preset under `configs/model/`
4. Add tests under `tests/models/`: forward shape, gradient flow, and if Taichi,
   forward/backward parity against the PyTorch reference
5. Update `README.md` if the architecture contract changes

Add or change a Taichi kernel:

1. Implement under `src/qwendopamine/kernels/taichi/`
2. Wrap forward/backward in a `torch.autograd.Function`
3. Expose through `src/qwendopamine/ops/`
4. Add parity tests in `tests/models/test_taichi_*.py`
5. Run focused Taichi tests plus `tests/ops/` and `tests/kernels/`

Add a new Hugging Face weight translation:

1. Edit `src/qwendopamine/models/infinidopamine/_qwen35_weights.py` and/or
   `_text_qwen35_weights.py`
2. Add a synthetic roundtrip test in
   `tests/models/test_hf_roundtrip_synthetic.py`
3. Add a slow 0.8B load test if practical

Add a training/eval feature:

1. Implement under `src/qwendopamine/training/` or `src/qwendopamine/evaluation/`
2. Wire into `loop.py`
3. Ensure checkpoint/schedule state round-trips through `schedules.py`
4. Add Hydra config under `configs/train/` or `configs/experiment/`
5. Add tests in `tests/models/test_training.py` / `test_schedules.py`

Notebook: `notebooks/train-infini-dopamine.ipynb` is Jupytext-paired with
`notebooks/train-infini-dopamine.py`. It downloads datasets and pushes
checkpoints to the Hub. Do not run unless asked.

## Boundaries

- Never placeholders, em-dashes, or silent scope expansion.
- Never revert or modify code you did not write unless explicitly asked.
- Ask before destructive commands, dependency upgrades, or scope expansion.
- Fix root causes; do not catch `KeyError`, `IndexError`, `TypeError`,
  `AttributeError` for developer bugs.
- Test behavior at public boundaries, not internal implementation.
