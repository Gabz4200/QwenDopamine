<div align="center">

# QwenDopamine

**A PyTorch research framework for Qwen-style LLM architectures with novel recurrent memory blocks, parallel reward branches, and Hugging Face / GGUF interoperability.**

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-managed-5c4ee5?style=flat-square)](https://docs.astral.sh/uv/)
[![HF Transformers](https://img.shields.io/badge/HuggingFace-transformers-ffd21e?style=flat-square)](https://huggingface.co/docs/transformers)
[![Taichi](https://img.shields.io/badge/Taichi-kernels-1f6feb?style=flat-square)](https://www.taichi-lang.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)

[Overview](#overview) • [Features](#features) • [Architecture](#architecture) • [Quick start](#quick-start) • [Python API](#python-api) • [Project layout](#project-layout) • [Testing](#testing)

</div>

---

## Overview

QwenDopamine re-implements Qwen-style LLM architectures. On top of a clean Qwen3.5 baseline it ships three research contributions:

- **Dual-stream Infini-attention**, which mixes a Gated DeltaNet-2 (GDN-2) linear recurrent memory with a local Sliding Window Attention (SWA), gated per-head by a learnable sigmoid projection.
- **Parallel Gated Reward Networks**, a `GatedRewardNet` fast-weight branch that runs alongside the main mixer with a near-zero init gate so pretrained backbones are preserved.
- **Optional Taichi-accelerated kernels** behind a single public ops layer (`qwendopamine.ops`) for the GDN-2 recurrence and the Reinforced-Delta memory core.

The framework runs on CPU out of the box (PyTorch CPU wheels are pinned via `uv` index), ships a pure-PyTorch training/eval harness, and loads pretrained Qwen3.5 weights with `strict=True`.

> [!NOTE]
> Research code, not a production model. Default upstream weights are `Qwen/Qwen3.5-0.8B` (see `qwendopamine.DEFAULT_QWEN35_REPO`).

## Features

- **Modular InfiniDopamine model** (`qwendopamine.models.infinidopamine`) with explicit per-layer mixer selection via `config.layer_types[layer_idx]`. No implicit block swaps based on neighbours.
- **Qwen3.5 weight compatibility**: pretrained text and vision checkpoints load directly with `strict=True`. Scalar erase gates expand across channel dimensions; gating parameters initialise to neutral defaults.
- **Multimodal vision tower** that inherits the Qwen3.5-VL encoder (`InfiniDopamineForConditionalGeneration`) for image + text continued pre-training, with a `merger` projection ready for LoRA.
- **Public ops layer** (`qwendopamine.ops`). Models import only from here. Forwards to Taichi kernels when available, otherwise falls back to pure-PyTorch references. No code path breaks when Taichi is absent.
- **Hydra-based training CLI** with separate config trees for `model/`, `train/`, `data/`, and `experiment/`. Schedule, freezing, and parallel-reward monitoring are first-class.
- **GGUF / safetensors / tokenizer integration** under `qwendopamine.integrations` for reading and writing the same model across HF, GGUF, and native PyTorch formats.
- **CPU-first local development** with `uv` lockfile, plus `[gpu]` and `[cpt]` extras for remote / notebook runs. No GPU required to read code, run the test suite, or load weights.

## Architecture

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │                InfiniDopamine Decoder Layer                   │
                       └─────────────────────────────────────────────────────────────┘
                                                      │
            ┌─────────────────────────────────────────┴──────────────────────────────────────────┐
            ▼                                                                                    ▼
   ┌────────────────────┐                                                              ┌────────────────────┐
   │     Main Mixer     │                                                              │   Parallel Branch  │
   │  (explicit per     │                                                              │   (opt-in reward)  │
   │   layer_types[])   │                                                              │                    │
   ├────────────────────┤                                                              ├────────────────────┤
   │ linear_attention / │                                                              │ InfiniDopamine     │
   │ gdn2 → GDN-2 + SWA │                                                              │ GatedRewardNet     │
   │ full / sliding →   │                                                              │  ├─ Memory Core    │
   │   Sliding Window   │                                                              │  ├─ Advantage Gate │
   │ gated_reward_net → │                                                              │  ├─ Value Baseline │
   │   GatedRewardNet   │                                                              │  └─ FiLM Modulator │
   └─────────┬──────────┘                                                              └─────────┬──────────┘
             │                                                                         │
             ▼                                                                         ▼
   shared input RMSNorm                                                RMSNorm → σ(W x + b) gate (init ≈ 0)
             │                                                                         │
             └─────────────────────  +  α · reward_out  ─────────────────────────────────┘
```

The main mixer is selected explicitly by `config.layer_types[layer_idx]`. The reward branch is **never** implicitly swapped in for the main mixer; it attaches only when the layer is listed in `config.parallel_reward_layers`, or when `use_parallel_reward=True` and the layer is an attention-only layer.

Two key components:

- **`InfiniDopamineGatedDeltaNet`** computes two attention representations from shared $Q, K, V$: a GDN-2 linear recurrent memory with decoupled channel-wise erase / write gates, and a local SWA over a causal window (default `W=1024`). A learnable per-head gate $\text{sigmoid}(\beta + W x_t)$ blends them, starting at $\text{sigmoid}(\beta)=0.5$ and learning per-token balance over training.
- **`InfiniDopamineGatedRewardNet`** runs alongside the main mixer with $h_L = h_{L-1} + \text{RMSNorm}(f_\text{main}(\cdot)) + \alpha \cdot \text{RMSNorm}(f_\text{dopamine}(\cdot), r)$. The gate $\alpha = \sigma(W x + b)$ is zero-initialised with bias $-5$ so $\sigma(-5) \approx 0.0067$ at start. Reward state (`recurrent_state`, `value_baseline`, `conv_state`) is written into `DynamicCache` under reward-specific keys so the GDN-2 cache is never clobbered.

Taichi-backed ops live in `qwendopamine.kernels.taichi` and are exposed through:

| Public op                                  | Backend                                  |
|--------------------------------------------|------------------------------------------|
| `qwendopamine.ops.gdn2.recurrent_taichi_gdn2`  | per-token recurrent GDN-2 (fwd + VJP)   |
| `qwendopamine.ops.gdn2.chunk_taichi_gdn2`      | chunkwise WY-style GDN-2 forward        |
| `qwendopamine.ops.reward.delta_core_step`      | Reinforced-Delta memory core (per-token)|
| `qwendopamine.ops.reward.chunkwise_delta_core_step_out` | chunkwise path with replay adjoint  |

See [AGENTS.md](AGENTS.md) for the full Taichi invariants (backend init in `runtime.py`, per-token replay for chunkwise adjoint, scratch buffer reuse, fp32 default).

## Quick start

The local CPU environment is the supported dev environment. Taichi resolves to its CPU backend when no GPU is available, so no code path breaks.

```bash
# Clone
git clone https://github.com/Gabz42/QwenDopamine.git
cd QwenDopamine

# Sync CPU + dev + HF extras (default quality-gate env)
uv sync --extra cpu --extra dev --extra hf

# Run the test suite (skip slow tests that download HF weights)
uv run pytest -m "not slow" -v

# Lint + type check
uv run ruff check .
uv run pyrefly check

# Train via Hydra CLI
uv run src/qwendopamine/cli/train.py
# or via the console script
uv run qwendopamine
```

Optional extras:

```bash
uv sync --extra gpu    # trl, unsloth
uv sync --extra cpt    # notebooks, jupyterlab, jupytext, peft, Pillow
```

> [!IMPORTANT]
> `import qwendopamine` is intentionally cheap (~0.0 s). The `qwendopamine.models` package uses PEP 562 lazy `__getattr__` and only loads model submodules on first access. Replacing this with eager imports costs ~12 s of cold start.

## Python API

```python
import torch
from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineForConditionalGeneration,
    InfiniDopamineTextConfig,
    InfiniDopamineVisionConfig,
)
from transformers import AutoProcessor, AutoTokenizer

# 1. Build a multimodal config mirroring Qwen3.5-0.8B
text_cfg = InfiniDopamineTextConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    sliding_window=1024,
)
vision_cfg = InfiniDopamineVisionConfig(
    hidden_size=768,
    out_hidden_size=1024,
    num_position_embeddings=2304,
)
cfg = InfiniDopamineConfig(text_config=text_cfg, vision_config=vision_cfg)
model = InfiniDopamineForConditionalGeneration(cfg)

# 2. Load pretrained Qwen3.5 weights (text + vision tower)
model.load_qwen35_weights("Qwen/Qwen3.5-0.8B", strict=True)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

# 3. Forward with optional reward_values for continued pre-training
input_ids = torch.tensor([[10, 20, 30, 40]])
reward_values = torch.zeros_like(input_ids, dtype=torch.float32)
output = model(input_ids=input_ids, reward_values=reward_values)
```

## Multimodal continued pre-training

[`notebooks/train-infini-dopamine.ipynb`](notebooks/train-infini-dopamine.ipynb) (Jupytext-paired `.py` companion) is a full multimodal CPT pipeline. It:

- Streams 16 HF datasets with per-dataset schema formatters (SMB frames, maze traces, sokoban CoT, chess PGN, ALFWorld trajectories, etc.).
- Uses `interleave_datasets` for memory-efficient mixing.
- Trains with LoRA on both text and vision tower projections.
- Saves merged checkpoints and pushes to the Hugging Face Hub.

Requires the `[cpt]` and `[hf]` extras:

```bash
uv sync --extra cpt --extra hf
```

> [!WARNING]
> That notebook downloads large datasets and pushes checkpoints to the Hub. **Do not run unless asked.**

## Project layout

```
QwenDopamine/
├── src/qwendopamine/
│   ├── models/
│   │   ├── infinidopamine/    # InfiniDopamine architecture, decoder, multimodal model
│   │   ├── qwen35/            # Qwen3.5 baseline + weight mappings
│   │   ├── gdn2/              # pure-PyTorch GDN-2 references + GatedRewardNet
│   │   ├── reinforced/        # Reinforced-Delta canonical reference
│   │   ├── blocks/            # block registry (BLOCKS) + reward components
│   │   └── core/              # RMSNorm, embeddings, LMHead
│   ├── ops/                   # public ops layer (models import from here only)
│   ├── kernels/taichi/        # Taichi-accelerated GDN-2 + Reinforced-Delta kernels
│   ├── integrations/          # HF, GGUF, safetensors, tokenizer loaders
│   ├── training/              # loop, schedules, freezing, metrics, parallel reward
│   ├── evaluation/            # perplexity, generation, layerwise stats
│   ├── cli/                   # Hydra training entrypoint
│   ├── distributed/           # FSDP / single-GPU helpers
│   └── utils.py
├── configs/                   # Hydra hierarchy: model/, train/, data/, experiment/
├── tests/                     # pytest suite (InfiniDopamine, GDN-2, reward, Qwen3.5)
├── notebooks/                 # Jupytext-paired CPT notebook
├── assets/                    # metric curves, images
└── AGENTS.md                  # working notes for coding agents (architecture, invariants)
```

## Testing

Pytest, config in `pyproject.toml`. Marker `slow` deselects tests that download Qwen3.5-0.8B weights.

```bash
# Full suite
uv run pytest -v

# Fast feedback (skip slow)
uv run pytest -m "not slow" -v

# Subsystem focus
uv run pytest tests/models/test_gdn2.py -v
uv run pytest tests/models/test_taichi_gdn2.py tests/models/test_taichi_gdn2_backward.py -v
uv run pytest tests/models/test_reward.py tests/models/test_rewardnet_canonical_validation.py -v
uv run pytest tests/ops/ -v
uv run pytest tests/kernels/ -v
```

Taichi tests gate on `qwendopamine.kernels.taichi.is_available()` and skip cleanly on machines where Taichi cannot initialise (headless CI, no GPU, missing drivers). To confirm the resolved backend:

```bash
uv run python -c "from qwendopamine.kernels.taichi import taichi_arch; print(taichi_arch())"
# expected: cpu (or cuda / gpu); 'unavailable' means the import or init failed
```

## Development conventions

- Standard HF `transformers` for tokenisation, generation, `DynamicCache`, and checkpoint publishing.
- Pure PyTorch for novel research blocks, altered residual connections, state dynamics, custom losses, and training loops.
- `r"""..."""` docstrings on public functions, matching the existing style in `cli/train.py` and `__init__.py`.
- No `print` for diagnostics in library code; use `tqdm` or `logging`.
- Imports: standard library, third-party, local; let `ruff` enforce.
- Keep `logs/`, `data/`, `checkpoints/`, `.venv/`, and `.aislop/` out of Git history.

Before opening a PR, run:

```bash
uv run ruff check .
uv run pyrefly check
uv run pytest -m "not slow" -v
```

## References

- Gated DeltaNet-2: [arXiv:2605.22791](https://arxiv.org/abs/2605.22791)
- Infini-attention: [arXiv:2404.07143](https://arxiv.org/abs/2404.07143)
- Hugging Face `transformers`: [docs](https://huggingface.co/docs/transformers)
- Taichi: [taichi-lang.org](https://www.taichi-lang.org/)