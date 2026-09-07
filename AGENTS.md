# AGENTS.md

Working notes for coding agents in the **QwenDopamine** repository. The closest
file in the directory tree wins over `/home/gabz/.agents/AGENTS.md`; respect that.

## What this project is

A PyTorch research framework that re-implements Qwen-style LLM architectures
with novel blocks (dual-stream Infini-attention, Gated Reward Net, GDN-2,
sliding window attention) and ships a pure-PyTorch training/eval harness with
Hugging Face `transformers` and GGUF weight interop.

- Package: `qwendopamine` (see `src/qwendopamine/__init__.py`)
- Default upstream weights: `Qwen/Qwen3.5-0.8B` (constant `DEFAULT_QWEN35_REPO`)
- Python: `>=3.12,<3.14` (pinned via `.python-version` → `3.13`); lockfile is `uv.lock`
- Backends supported: CPU (default) and GPU (`[gpu]` extra); Taichi kernels for
  GDN-2 and `GatedRewardNet` are tested via `tests/models/test_taichi_*`

## Setup

Use `uv` (lockfile committed). The CPU-only local env is the supported dev env.

```bash
uv sync --extra cpu --extra dev --extra hf
uv run pytest -v
uv run pyrefly check
uv run ruff check .
```

GPU / CPT extras exist but are not needed for normal code/test work:

```bash
uv sync --extra gpu        # trl, unsloth
uv sync --extra cpt        # notebooks, jupyterlab, jupytext, peft, Pillow
```

There is no `Makefile`, no `.github/`, no `CI` config in-tree. Local uv
commands ARE the quality gate.

## Layout (where to put things)

- `src/qwendopamine/`: package
  - `models/infinidopamine/`: modular InfiniDopamine architecture, decoder
    layer, multimodal model, weight mappings (`_qwen35_weights.py`,
    `_text_qwen35_weights.py`)
  - `models/qwen35/`: modular Qwen3.5 baseline + weight mappings
  - `models/gdn2/`: pure-PyTorch reference kernels (`torch_chunk_gdn2`,
    `torch_recurrent_gdn2`) and `GatedRewardNet`
  - `models/blocks/`: block registry (`BLOCKS`) and reward components
  - `models/core/`: light core primitives (`RMSNorm`, embeddings, `LMHead`)
  - `models/reinforced/`, `models/shared/`, `models/gdn2_gpt/`
  - `integrations/`: HF, GGUF, safetensors, tokenizer loaders
  - `training/`: `loop.py`, `schedules.py`, `freezing.py`, `metrics.py`,
    `parallel_reward.py`
  - `evaluation/`: perplexity, generation, layerwise stats
  - `cli/train.py`: Hydra entrypoint; `__main__.py` is `qwendopamine:main`
  - `ops/`: public operations layer (the only API models may import). Forwards
    to Taichi kernels when available, else pure-PyTorch references.
  - `kernels/taichi/`: Taichi-accelerated GDN-2 and Reinforced-Delta kernels.
    See the **Taichi kernels** section below for the full contract.
  - `distributed/`, `utils.py`
- `configs/`: Hydra hierarchy. **Never** edit by hand-patching without running
  `python -c "import hydra; hydra.compose('train/cpu')"` to validate the override graph
  - `configs/model/{base,infinidopamine_*,qwen35_*}.yaml`
  - `configs/train/{cpu,single_gpu,fsdp2,frozen_backbone,*_frozen_backbone}.yaml`
  - `configs/experiment/{ablation_block_position,ablation_unfreezing}.yaml`
- `tests/`: pytest suite (see Testing below)
- `notebooks/train-infini-dopamine.ipynb` (+ `.py` jupytext pair): multimodal CPT pipeline
- `assets/`, `annotations/`, `outputs/`: auxiliary data, ignored from git

### Lazy-import contract

`qwendopamine.models.__init__` uses **PEP 562 lazy `__getattr__`** so
`import qwendopamine.models` is ~0.0s. Touching a model class (e.g.
`qwendopamine.models.qwen35.Qwen35Config`) imports the underlying submodule
on first access. Don't replace this with eager imports. Measured cost is
~12s of cold start without it.

## Taichi kernels

The Taichi layer is a first-class engine (not a prototype) for two hot paths:
the GDN-2 recurrence (`chunk_taichi_gdn2`, `recurrent_taichi_gdn2`) and the
Reinforced-Delta memory core (`delta_core_step`, `delta_core_step_out`). It
sits behind the public `qwendopamine.ops` package: **models MUST import from
`qwendopamine.ops`, never from `qwendopamine.kernels.taichi.*`**.

### Backend init

Initialised lazily in `kernels/taichi/runtime.py` on first call. The contract:

1. `ti.init(arch=ti.gpu, default_fp=ti.f32)` is tried first. Taichi itself
   picks CUDA, then Vulkan, then Metal/OpenGL, then CPU.
2. If that fails (headless box, missing GPU drivers), a second
   `ti.init(default_fp=ti.f32)` is tried (Taichi's own default priority).
3. The resolved arch is cached in `_ARCH`; every kernel compile then targets
   the same backend. Callers can read it via
   `qwendopamine.kernels.taichi.taichi_arch()`, which returns one of
   `"cpu"`, `"cuda"`, `"gpu"`, `"unavailable"`.
4. `taichi>=1.7.4` is in `[project.dependencies]` (not in an extra). The
   CPU fallback works on a stock `uv sync --extra cpu` install.
5. If Taichi cannot be imported at all, `_HAS_TAICHI = False` and every op
   falls back to the pure-PyTorch reference in `models/gdn2/recurrence/*` and
   `models/reinforced/canonical_reference.py`. No code path should break.

### Recurrent GDN-2 (per-token path)

- Public entry: `qwendopamine.ops.gdn2.recurrent_taichi_gdn2` (and
  `chunk_taichi_gdn2`).
- Forward: per-token VJP-compatible kernel
  (`kernels/taichi/gdn2_kernels.launch_recurrent_step`). Same BTHD layout
  and calling convention as `torch_recurrent_gdn2` /
  `triton.fused_recurrent_gdn2`.
- Backward: per-token VJP replay in reverse over the saved per-token states
  (`launch_recurrent_step_bwd`). Do NOT hand-write a chunked WY-bwd kernel.
  The replay pattern matches the FLA library's Taichi backend.
- PyTorch `torch.autograd.Function` wrapper: `_RecurrentTaichiGdn2Function`
  in `kernels/taichi/_recurrent_path.py`.

### Chunkwise GDN-2 (WY-style path)

- Public entry: `qwendopamine.ops.gdn2.chunk_taichi_gdn2`.
- Forward: Taichi WY-style kernel
  (`launch_chunk_fwd_per_bh` in `kernels/taichi/gdn2_kernels.py`).
- Backward: re-runs the equivalent torch reference in reverse to obtain
  per-input gradients. This is intentional. See the comment block at the top
  of `gdn2_api.py`. The Taichi forward is the production engine; the
  numerical agreement holds because both paths operate on identical inputs
  and identical mathematical algorithm.
- `torch.autograd.Function` wrapper: `_ChunkTaichiGdn2Function` in
  `kernels/taichi/_chunk_path.py`.

### Reinforced Delta (RewardNet) kernels

- Public entries: `qwendopamine.ops.reward.delta_core_step` (functional),
  `delta_core_step_out` (in-place). Used by `GatedRewardNet` /
  `InfiniDopamineGatedRewardNet`.
- Per-token fwd/bwd: `launch_delta_core_step` /
  `launch_delta_core_step_bwd` in
  `kernels/taichi/reinforced_kernels.py`. Wrapped by
  `_DeltaCoreStepFunction`.
- Chunkwise path: `_ChunkwiseDeltaCoreStepFunction`, exported as
  `chunkwise_delta_core_step_out`. The chunkwise backward replays the
  per-token VJP via `launch_chunk_bwd_per_bh`.
- The public per-channel gate helper
  `_make_effective_gate(omega, gate) -> [B, D]` is the single contraction
  point for `omega * write`. **Use it.** Do not hand-roll
  `omega_w_eff = omega_w * write` at any callsite.

### Scratch buffers

`_SCRATCH` in `kernels/taichi/gdn2_kernels.py` is a `(C, K, V)` keyed cache
of per-chunk scratch tensors. Lazy on first invocation, reused across calls.
Python-side callers never touch it; the launch wrappers pass it as kernel
arguments.

### Tests and skip patterns

All Taichi tests gate on `qwendopamine.kernels.taichi.is_available()` and
will skip cleanly when Taichi cannot initialise (headless CI, no GPU, etc.):

| Path | Covers |
| --------------------------------------- | ----------------------------------------------- |
| `tests/models/test_taichi_gdn2.py` | GDN-2 recurrent + chunkwise forward numerics |
| `tests/models/test_taichi_gdn2_backward.py` | GDN-2 fwd/bwd parity vs torch reference |
| `tests/models/test_taichi_rewardnet.py` | Reinforced-Delta forward + `launch_delta_core_step` |
| `tests/models/test_taichi_rewardnet_backward.py` | Reinforced-Delta backward parity |
| `tests/models/test_reinforced_delta_layer_taichi_path.py` | Taichi vs torch end-to-end via `ReinforcedDeltaLayer`; verifies `use_taichi=True` routes through the Taichi autograd `Function` |
| `tests/kernels/test_kernels_module_importable.py` | Module surface + `taichi_arch` reporting |

Run them directly:

```bash
uv run pytest tests/models/test_taichi_gdn2.py -v
uv run pytest tests/models/test_taichi_gdn2_backward.py -v
uv run pytest tests/models/test_taichi_rewardnet.py tests/models/test_taichi_rewardnet_backward.py -v
uv run pytest tests/models/test_reinforced_delta_layer_taichi_path.py -v
uv run pytest tests/kernels/ -v
```

To check which backend Taichi resolved to:

```bash
uv run python -c "from qwendopamine.kernels.taichi import taichi_arch; print(taichi_arch())"
# expected: cpu (or cuda / gpu); 'unavailable' means the import or init failed
```

### Invariants (the "don't break these" list for Taichi)

- `kernels/taichi/runtime.py` is the **only** place that calls `ti.init()`.
  No kernel file may call it directly; if you do, the cached `_ARCH` is
  bypassed and compile targets can disagree.
- Per-token replay is the only safe chunkwise adjoint pattern. Do not
  write a WY-bwd kernel. Replay the forward N VJP steps in reverse with
  `dstate` rolled.
- Public ops in `qwendopamine.ops.*` are the model-facing contract; the
  kernel modules may change freely without touching models.
- Numerical agreement between Taichi and torch reference is the test
  contract. If you change a kernel, the corresponding `test_taichi_*` test
  must still pass against the unchanged torch reference.
- Default fp is `ti.f32`. Do not switch to `ti.f64` for a single kernel.
  It would silently disagree with the torch reference on reductions.

## Setup & build

```bash
# Local CPU env (dev default)
uv sync --extra cpu --extra dev --extra hf

# Run the Hydra training CLI
uv run src/qwendopamine/cli/train.py
# or via the console script
uv run qwendopamine
```

Hydra config search path is rooted at `configs/` with default
`configs/train/cpu.yaml`. Override examples:

```bash
uv run src/qwendopamine/cli/train.py model=infinidopamine_reference
uv run src/qwendopamine/cli/train.py train=single_gpu model=qwen35_gdn2
uv run src/qwendopamine/cli/train.py +experiment=ablation_unfreezing
```

## Testing

Pytest, config in `pyproject.toml`. `testpaths = ["tests"]`, marker `slow`
(deselect with `-m "not slow"`).

```bash
# Full suite
uv run pytest -v

# Skip slow (downloads HF weights, runs Qwen3.5-0.8B end-to-end)
uv run pytest -m "not slow" -v

# Subsystem focus
uv run pytest tests/models/test_gdn2.py -v
uv run pytest tests/models/test_taichi_gdn2.py tests/models/test_taichi_gdn2_backward.py -v
uv run pytest tests/models/test_reward.py tests/models/test_rewardnet_canonical_validation.py -v
uv run pytest tests/ops/ -v
uv run pytest tests/kernels/ -v

# Single test by node id
uv run pytest tests/models/test_infinidopamine.py::test_xxx -v
```

Test categories (non-exhaustive):

| Path                                            | Covers                                          |
| ----------------------------------------------- | ----------------------------------------------- |
| `tests/models/test_gdn2*`                       | GDN-2 reference kernels, decomposition, params  |
| `tests/models/test_taichi_gdn2*.py`             | Taichi backend, forward + backward              |
| `tests/models/test_reward*.py`                  | `GatedRewardNet`, canonical validation          |
| `tests/models/test_taichi_rewardnet*.py`        | Taichi rewardnet forward + backward             |
| `tests/models/test_infinidopamine*.py`          | Modular InfiniDopamine model + HF weight load   |
| `tests/models/test_qwen35*.py`                  | Qwen3.5 reference + 0.8B HF roundtrip (slow)    |
| `tests/models/test_parallel_reward_monitoring.py` | Parallel reward branch monitoring             |
| `tests/models/test_reinforced_delta_layer_taichi_path.py` | Taichi reinforced-delta end-to-end  |
| `tests/models/test_schedules.py`                | LR schedules + freezing logic                   |
| `tests/models/test_training.py`                 | Training loop                                   |
| `tests/models/test_safetensors.py`              | Checkpoint roundtrip                            |
| `tests/models/test_integrations.py`             | HF / GGUF integration paths                     |
| `tests/ops/*`                                   | Op dispatch + device routing + autograd         |
| `tests/kernels/*`                               | Kernel module importability                     |

## Code style & quality gates

- **Formatter / linter**: `ruff` (no config file; defaults are intentional)
- **Type checker**: `pyrefly` with `pyrefly.toml` (single global config). Tuned
  to surface real type errors and unannotated public surface, while silencing
  inference noise from third-party model code (HF `transformers`, Taichi
  dynamic runtime, HF kwargs splat).
- Always run **before commit**:

  ```bash
  uv run ruff check .
  uv run pyrefly check
  uv run pytest -m "not slow" -v
  ```

- Python: 3.12+ syntax (`from __future__ import annotations` everywhere in
  `src/`). Type-annotate all public APIs. Use `r"""..."""` docstrings on
  public functions (matches existing style in `cli/train.py`, `__init__.py`).
- No `print` for diagnostics in library code; use `tqdm` or `logging`.
- Imports: standard library, third-party, local; let `ruff` enforce.

## Architecture & invariants (the "don't break these" list)

1. **Layer-type selection is explicit.** No block is swapped in implicitly based
   on neighbouring layers. Main mixer per layer is `config.layer_types[layer_idx]`.
2. **Parallel reward branch never replaces the main mixer.** It is additive
   only, attached when `config.parallel_reward_layers` lists the index OR
   (`use_parallel_reward=True` AND layer is `full_attention` / `sliding_attention`).
   The gate is initialised so `sigmoid(bias) ≈ 0.0067` so pretrained backbones
   are left effectively untouched at start.
3. **Reward state persistence is explicit.** `GatedRewardNet.forward` returns
   `{"recurrent_state", "value_baseline", "conv_state"}`, and the
   `InfiniDopamineGatedRewardNet` wrapper writes them into `DynamicCache`
   under reward-specific keys (`reward_recurrent_state`, `reward_value_baseline`,
   `reward_conv_states`). Never clobber GDN-2 cache.
4. **GDN-2 gate shape contract.** Per-channel weights (`omega_w`, `write`,
   `erase`) live inside the einsum, not after it. `dk[k] = sum_d omega_W[d] *
   dS[d,k] * e[d]` is `einsum("bd,bdk,bk->bk", omega_W, dS, e)`. Do NOT
   broadcast a `[B, D]` × `[B, K]` outer product into `[B, D, K]` upstream.
5. **Chain rule is multiplicative, not division.** Given
   `omega_w_eff = omega_w * write`, recover `d_write = d_omega_w_eff * omega_w`.
   Division by `write.clamp_min(1e-12)` is wrong and numerically fragile.
6. **Taichi per-token replay for chunkwise adjoint.** The backward is the
   forward N per-token VJP calls in reverse with `dstate` rolled. Do not
   hand-write a chunked WY-bwd kernel. See the **Taichi kernels** section
   for the full contract (backend init, scratch buffers, public ops).
7. **Low-precision numerics.** Upcast to `float32` before `.pow()`, `.sum()`,
   `.mean()` on `float16` / `bfloat16` tensors.
8. **Optimizer / scheduler / global-step are guarded together.** Skipped steps
   (GradScaler skip) must not advance any training state. Schedulers must
   serialise `step_count`, `phase`, `warmup_steps` and re-hydrate on restore.
9. **HF config class hierarchy.** Multimodal configs nest `text_config` and
   `vision_config`; never read top-level `hidden_size` on
   `InfiniDopamineConfig`. Go through `text_config`.
10. **Experiment YAMLs validate registered block names at load time.** An
    unregistered block name breaks Hydra instantiation. Update
    `src/qwendopamine/models/blocks/registry.py` (`BLOCKS`) first.
11. **Qwen3.5 weight loading is `strict=True`.** Scalar erase gates expand
    across channel dims `d_k` and `d_v`; any new param must have a translation
    rule in `_qwen35_weights.py` / `_text_qwen35_weights.py`.

## Common tasks

### Add a new block

1. Implement under `src/qwendopamine/models/blocks/` (or `models/reinforced/`
   if it's a GDN-2 derivative).
2. Register it in `src/qwendopamine/models/blocks/registry.py` (`BLOCKS` dict).
3. Add a Hydra preset under `configs/model/` (e.g.
   `infinidopamine_<name>.yaml`).
4. Add tests under `tests/models/`. At minimum: forward shape, gradient
   flow, and (if Taichi) forward+backward parity with the PyTorch reference.
5. Update `README.md` (architecture section) and the architecture overview
   above if the contract changes.

### Add or change a Taichi kernel

1. Implement the kernel under `src/qwendopamine/kernels/taichi/` (extend the
   relevant module: `gdn2_kernels.py`, `reinforced_kernels.py`, or a new
   sibling). Do not call `ti.init()` here; let `kernels/taichi/runtime.py`
   own init.
2. Wrap forward + backward in a `torch.autograd.Function`. Use the per-token
   replay pattern for any chunkwise adjoint; do not write a WY-bwd kernel.
3. Expose the new op through `src/qwendopamine/ops/`. This is the public
   contract. Update `qwendopamine.kernels.taichi.__init__` only if the
   kernel module also needs a re-export.
4. Add a parity test under `tests/models/test_taichi_*.py` that compares
   numerics against the pure-PyTorch reference in `models/gdn2/recurrence/*`
   or `models/reinforced/canonical_reference.py`. Skip pattern:
   `pytest.mark.skipif(not is_available(), ...)`.
5. Run the focused Taichi test file plus `tests/ops/` and `tests/kernels/`
   before opening a PR. Verify the resolved backend with
   `python -c "from qwendopamine.kernels.taichi import taichi_arch; print(taichi_arch())"`.

### Add a new Hugging Face weight translation

1. Edit `src/qwendopamine/models/infinidopamine/_qwen35_weights.py` and/or
   `_text_qwen35_weights.py`. Maintain an explicit GGUF → HF translation
   layer; do not assume 1:1 key mapping.
2. Add a synthetic roundtrip test in
   `tests/models/test_hf_roundtrip_synthetic.py` and (if practical) a
   slow `Qwen/Qwen3.5-0.8B` load test in
   `tests/models/test_infinidopamine_hf_loading.py` or
   `tests/models/test_qwen35_hf_08b.py`.

### Add a training/eval feature

1. Implement under `src/qwendopamine/training/` (or `evaluation/`).
2. Wire it into `loop.py` (single entrypoint; do not fork).
3. If it has schedule or checkpoint state, ensure it round-trips through
   `schedules.py` and serialises the full set of counters
   (`step_count`, `phase`, `warmup_steps`, plus anything new).
4. Add a Hydra config under `configs/train/` or `configs/experiment/`.
5. Tests in `tests/models/test_training.py` / `test_schedules.py`.

### Run the multimodal CPT notebook

`notebooks/train-infini-dopamine.ipynb` is a Jupytext-paired notebook
(`.ipynb` + `.py`). To execute end-to-end you need the `[cpt]` extra (notebooks,
jupytext, jupyterlab, peft, Pillow, trl, unsloth) and the `[hf]` extra
(datasets, tokenizers). It downloads 16 HF datasets and pushes merged
checkpoints to the Hub. **Do not run unless asked.**

## Boundaries (per `/home/gabz/.agents/AGENTS.md`)

- Never placeholders, em-dashes, or silent scope expansion.
- Never revert or modify code you did not write unless the user explicitly asks.
- Ask first before: destructive commands, deleting code you didn't write,
  dependency upgrades, scope expansion beyond the user's stated request.
- Produce evidence (pasted test output, exact command + exit code, screenshot,
  or short transcript) before claiming a non-trivial change is done.
- Follow `code-quality` and `ml-implementation-preferences` domain rules
  (auto-loaded for `**/*.py`).
- Match `testing` rules for `tests/`. Test outcomes at public boundaries,
  not internal implementation; no plumbing/source-text/incidental-default
  assertions.
- Match `uv-dependency-management` rules when touching `uv.lock` /
  `pyproject.toml`.

## What is intentionally NOT here

- A CI workflow file (`.github/`). The project is run locally via uv.
- A `Makefile` / `justfile`. uv scripts ARE the entrypoints.
- Ruff configuration. Defaults are used; lint rules live in `ruff` itself.
- Dockerfiles / deploy scripts. This is a research framework, run in a
  notebook or local Python env.