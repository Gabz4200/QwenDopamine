<div align="center">

# QwenDopamine

**PyTorch research framework for Qwen-style LLMs with recurrent memory blocks, parallel reward branches, and Hugging Face / GGUF interoperability.**

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-managed-5c4ee5?style=flat-square)](https://docs.astral.sh/uv/)
[![HF Transformers](https://img.shields.io/badge/HuggingFace-transformers-ffd21e?style=flat-square)](https://huggingface.co/docs/transformers)
[![Taichi](https://img.shields.io/badge/Taichi-kernels-1f6feb?style=flat-square)](https://www.taichi-lang.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)

[Quick start](#quick-start) • [Python API](#python-api) • [Architecture](#architecture) • [Configs](#configuration) • [Testing](#testing)

</div>

> [!NOTE]
> Research code, not a production model. Default upstream weights are [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B).

## Why QwenDopamine

Qwen-style transformers handle local context well but pay quadratic cost for long-horizon memory. QwenDopamine tests three complementary approaches:

| Contribution | What it does |
|---|---|
| **Dual-stream Infini-attention** | Mixes Gated DeltaNet-2 (GDN-2) linear recurrent memory with Sliding Window Attention (SWA), gated per head by a learnable sigmoid projection |
| **Gated Reward Net** | Parallel reward branch with persistent recurrent, value, and conv state written into `DynamicCache` under reward-specific keys, additive only and never replacing the main mixer |
| **Taichi-accelerated kernels** | Forward and backward kernels for GDN-2 and Reinforced-Delta with automatic fallback to pure-PyTorch references when Taichi is unavailable |

Also included: Hugging Face and GGUF weight interop, Hydra-driven training configs, and a multimodal continued pre-training (CPT) pipeline that streams 16 datasets.

## Quick start

### Prerequisites

- Python `>=3.12,<3.14`
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
# CPU development environment (recommended for local work)
uv sync --extra cpu --extra dev --extra hf

# GPU (CUDA 12.8)
uv sync --extra gpu --extra dev --extra hf

# Multimodal CPT notebook extras
uv sync --extra cpt --extra hf
```

### Run the training CLI

```bash
# default config
uv run src/qwendopamine/cli/train.py

# via installed entrypoint
uv run qwendopamine
```

### Override configs from the CLI

```bash
uv run src/qwendopamine/cli/train.py model=infinidopamine_reference
uv run src/qwendopamine/cli/train.py train=single_gpu model=qwen35_gdn2
uv run src/qwendopamine/cli/train.py experiment=ablation_unfreezing
```

> [!IMPORTANT]
> `import qwendopamine` is intentionally cheap. The `qwendopamine.models` package uses PEP 562 lazy `__getattr__` and only loads model submodules on first attribute access. Do not replace this with eager imports.

## Python API

### Load a model from Hub weights

```python
from qwendopamine.models.infinidopamine import InfiniDopamineConfig, InfiniDopamineForCausalLM

config = InfiniDopamineConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
model = InfiniDopamineForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", config=config)
model.eval()
```

### Explicit per-layer mixer selection

Layer-type selection is explicit. No block is swapped implicitly based on neighbours. Use `config.layer_types[layer_idx]`:

```python
config.layer_types = ["full_attention", "sliding_attention", "full_attention"]
```

The block registry in `src/qwendopamine/models/blocks/registry.py` maps these names to implementations via `build_block(...)`. Experiment YAML names must match registered block names.

### Dynamic model factory

```python
from qwendopamine.models.model_factory import create_model, build_model

model = create_model("qwen35", config)
model = create_model("infinidopamine", config)
model = create_model("research", config)

# Hydra-aware: resolves family from config
model = build_model(config)
```

Built-in families: `qwen35`, `infinidopamine`, `research`. Register custom families with `register_model_family(name, builder)` (timm-style).

### Public ops layer

Models must import operations from `qwendopamine.ops`, never directly from `qwendopamine.kernels.taichi.*`:

```python
from qwendopamine.ops.gdn2 import chunk_taichi_gdn2, recurrent_taichi_gdn2
from qwendopamine.ops.reward import delta_core_step, delta_core_step_out
```

The Taichi runtime (`src/qwendopamine/kernels/taichi/runtime.py`) is the only place that calls `ti.init()` and selects the backend as CUDA > Vulkan > Metal/OpenGL > CPU. The pure-PyTorch reference is the fallback when Taichi cannot initialise.

### Hugging Face interop

```python
# Save and reload through transformers Auto* APIs
model.save_pretrained("./ckpt")
reloaded = InfiniDopamineForCausalLM.from_pretrained("./ckpt")

# GGUF export / import
from qwendopamine.integrations.gguf import save_as_gguf, load_from_gguf
```

## Architecture

```
src/qwendopamine/
├── models/               # model implementations
│   ├── infinidopamine/   # InfiniDopamine model, configs, HF weight translations
│   ├── qwen35/           # Qwen3.5 baseline
│   ├── gdn2/             # GDN-2 blocks, recurrence (chunk + recurrent paths)
│   ├── gdn2_gpt/         # GDN-2 GPT variant
│   ├── reinforced/       # GatedRewardNet / Reinforced-Delta references
│   ├── blocks/           # BlockRegistry + build_block(...)
│   ├── shared/           # shared heads, pretrained mixins, text/vision towers
│   └── core/             # embeddings, RMSNorm, LM head, config adapter
├── ops/                  # public operations layer (models import from here)
│   ├── gdn2.py           # chunk / recurrent GDN-2 dispatch
│   ├── reward.py         # delta core step dispatch
│   └── references/       # pure-PyTorch oracle / reference implementations
├── kernels/taichi/       # Taichi kernels (forward + backward via autograd.Function)
├── training/             # loop, schedules, freezing, parallel reward, metrics
├── evaluation/           # perplexity, generation, layerwise stats
├── integrations/         # HF, GGUF, safetensors, tokenizer loaders
├── distributed/          # distributed setup
└── cli/                  # Hydra entrypoint (train.py)
configs/                  # Hydra hierarchy (model / train / data / experiment)
tests/                    # pytest suite
notebooks/                # multimodal CPT notebook (Jupytext-paired)
```

**Dependency direction:**

```
models/* --> ops/* --> kernels/taichi/*
           |-> models/gdn2/recurrence/*  (pure-PyTorch reference)
           |-> models/reinforced/canonical_reference.py
```

**Key registries:**

| Registry | Location | Purpose |
|---|---|---|
| `BlockRegistry` | `models/blocks/registry.py` | Maps Hydra block names to layer implementations; lazy `_populate()` avoids circular imports |
| `BackendRegistry` | `ops/_backend_registry.py` | Dispatches GDN-2 / Reward ops between Taichi and PyTorch backends |
| `ModelRegistry` | `models/model_factory.py` | Dynamic model family loading via `register_model_family` / `create_model` |

## Configuration

Hydra configs live under `configs/`:

| Directory | Presets |
|---|---|
| `configs/model/` | `base.yaml`, `qwen35_reference`, `qwen35_gdn2`, `qwen35_custom_block`, `infinidopamine_reference`, `infinidopamine_gdn2`, `infinidopamine_custom_block` |
| `configs/train/` | `cpu.yaml`, `single_gpu.yaml`, `fsdp2.yaml`, `frozen_backbone.yaml`, `cpu_frozen_backbone.yaml` |
| `configs/data/` | `pretrain.yaml`, `debug.yaml` |
| `configs/experiment/` | `ablation_unfreezing.yaml`, `ablation_block_position.yaml` |

```bash
# Compose presets on the command line
uv run src/qwendopamine/cli/train.py model=qwen35_gdn2 train=cpu data=debug
```

## Multimodal continued pre-training

`notebooks/train-infini-dopamine.ipynb` (Jupytext-paired with `train-infini-dopamine.py`) is a full CPT pipeline. It streams 16 HF datasets with per-dataset schema formatters and pushes merged checkpoints to the Hub via `accelerate.PartialState()`.

```bash
uv sync --extra cpt --extra hf
# then open notebooks/train-infini-dopamine.ipynb
```

> [!WARNING]
> This notebook downloads large datasets and pushes checkpoints to the Hub. Do not run unless explicitly asked. Keep the `.py` / `.ipynb` pair in sync with `jupytext --sync`.

> [!TIP]
> On Kaggle, use `accelerate.PartialState()`, not `accelerate launch` / `torchrun`. Use `git+https` or archive zip URLs, bare .git URLs break pip.

## Testing

Pytest is configured in `pyproject.toml`. The `slow` marker covers tests that download Qwen3.5-0.8B weights.

```bash
# full suite
uv run pytest -v

# skip slow tests (default for local iteration)
uv run pytest -m "not slow" -v

# focused areas
uv run pytest tests/models/test_gdn2.py -v
uv run pytest tests/models/test_taichi_gdn2.py tests/models/test_taichi_gdn2_backward.py -v
uv run pytest tests/models/test_reward*.py -v
uv run pytest tests/models/test_reinforced_delta_layer_taichi_path.py -v
uv run pytest tests/ops/ -v
uv run pytest tests/kernels/ -v
```

Confirm the resolved Taichi backend:

```bash
uv run python -c "from qwendopamine.kernels.taichi import taichi_arch; print(taichi_arch())"
# expected: cpu, cuda, or gpu; 'unavailable' means Taichi cannot initialise
```

## Core rules

These invariants are load-bearing. Violations break training or weight loading:

- **Layer-type selection is explicit:** No block is swapped implicitly based on neighbouring layers.
- **Reward branch is additive only:** It never replaces the main mixer. It attaches only when listed in `config.parallel_reward_layers`, or when `use_parallel_reward=True` and the layer is attention-only.
- **Reward state persistence is explicit:** `GatedRewardNet.forward` returns recurrent, value, and conv state. The wrapper writes them into `DynamicCache` under reward-specific keys without clobbering GDN-2 cache.
- **GDN-2 gate shape contract:** Per-channel weights live inside the einsum: `einsum("bd,bdk,bk->bk", omega_W, dS, e)`. Do not broadcast a `[B, D]` x `[B, K]` outer product upstream.
- **Chain rule is multiplicative:** Given `omega_w_eff = omega_w * write`, recover `d_write = d_omega_w_eff * omega_w`. Do not divide by `write.clamp_min(1e-12)`.
- **Low-precision numerics:** Upcast to `float32` before `.pow()`, `.sum()`, `.mean()` on `float16` / `bfloat16`.
- **Effective gate contraction:** `_make_effective_gate(omega, gate) -> [B, D]` is the single contraction point for `omega * write`.
- **HF config hierarchy:** Multimodal configs nest `text_config` and `vision_config`. Never read top-level `hidden_size` on `InfiniDopamineConfig`; go through `text_config`.
- **Qwen3.5 weight loading uses `strict=True`:** New parameters need translation rules in `_qwen35_weights.py` / `_text_qwen35_weights.py`.
- **Taichi invariants:** `kernels/taichi/runtime.py` is the only place that calls `ti.init()`. Default float precision is `ti.f32`. Per-token replay is the only safe chunkwise adjoint pattern.

> [!TIP]
> See [`AGENTS.md`](AGENTS.md) for the full development workflows: adding blocks, kernels, weight translations, and training features.

## Quality gates

No `.github/`, no `Makefile`, no in-tree CI. Local `uv` commands are the quality gate. Run before committing:

```bash
uv run ruff check --fix .
uv run aislop scan
uv run pytest
uv run pyrefly check
```

- **ruff**, linter and formatter (default rules, `--fix` auto-fixes)
- **aislop**, agentic quality gate (must be 0 errors; warnings are medium-confidence)
- **pyrefly**, type checker (`pyrefly.toml` is committed; ~113 suppressed categories for HF/Taichi)
- **pytest**, full suite is the verification target; use `-m "not slow"` for quick focused runs

## References

- Gated DeltaNet-2: [arXiv:2605.22791](https://arxiv.org/abs/2605.22791)
- Taichi: [taichi-lang.org](https://www.taichi-lang.org/)
- Qwen3.5: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) on Hugging Face
