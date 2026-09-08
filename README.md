<div align="center">

# QwenDopamine

**A PyTorch research framework for Qwen-style LLM architectures with recurrent memory blocks, parallel reward branches, and Hugging Face / GGUF interoperability.**

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-managed-5c4ee5?style=flat-square)](https://docs.astral.sh/uv/)
[![HF Transformers](https://img.shields.io/badge/HuggingFace-transformers-ffd21e?style=flat-square)](https://huggingface.co/docs/transformers)
[![Taichi](https://img.shields.io/badge/Taichi-kernels-1f6feb?style=flat-square)](https://www.taichi-lang.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)

</div>

## What it does

QwenDopamine re-implements Qwen-style LLMs with three research contributions:

- **Dual-stream Infini-attention**. Mixes Gated DeltaNet-2 (GDN-2) linear recurrent memory with local Sliding Window Attention (SWA), gated per-head by a learnable sigmoid projection.
- **Gated Reward Net**. Parallel reward branch with persistent recurrent/value/conv state written into `DynamicCache` under reward-specific keys.
- **Taichi-backed kernels**. GDN-2 and Reinforced-Delta kernels that fall back to pure-PyTorch references when Taichi is unavailable.

Default upstream weights are `Qwen/Qwen3.5-0.8B`.

> [!NOTE]
> Research code, not a production model.

## Quick start

```bash
# CPU dev environment (default)
uv sync --extra cpu --extra dev --extra hf
```

Run the Hydra training CLI:

```bash
uv run src/qwendopamine/cli/train.py
uv run qwendopamine
```

Override configs from the CLI:

```bash
uv run src/qwendopamine/cli/train.py model=infinidopamine_reference
uv run src/qwendopamine/cli/train.py train=single_gpu model=qwen35_gdn2
```

> [!IMPORTANT]
> `import qwendopamine` is intentionally cheap (~0.0s). The `qwendopamine.models` package uses PEP 562 lazy `__getattr__` and only loads model submodules on first access. Do not replace this with eager imports.

## Python API

```python
import torch
from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineForCausalLM,
)

config = InfiniDopamineConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
model = InfiniDopamineForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", config=config)

# Explicit per-layer mixer selection
config.layer_types = ["full_attention", "sliding_attention", "full_attention"]

# Use Taichi-backed ops when available
from qwendopamine.ops.gdn2 import chunk_taichi_gdn2, recurrent_taichi_gdn2
from qwendopamine.ops.reward import delta_core_step, delta_core_step_out
```

## Multimodal continued pre-training

`notebooks/train-infini-dopamine.ipynb` (Jupytext-paired `.py` companion) is a multimodal CPT pipeline. It streams 16 HF datasets with per-dataset schema formatters and pushes merged checkpoints to the Hub.

```bash
uv sync --extra cpt --extra hf
```

> [!WARNING]
> This notebook downloads large datasets and pushes checkpoints to the Hub. Do not run unless asked.

## Project layout

```
src/qwendopamine/
├── models/             # model implementations
│   ├── infinidopamine/ # InfiniDopamine model, configs, HF weight translations
│   ├── qwen35/         # Qwen3.5 baseline
│   ├── gdn2/           # GDN-2 reference kernels
│   ├── reinforced/     # GatedRewardNet / Reinforced-Delta references
│   └── blocks/         # block registry
├── ops/                # public operations layer (models import from here)
├── kernels/taichi/     # Taichi kernels for GDN-2 and Reinforced-Delta
├── training/           # training loop, schedules, parallel reward
├── evaluation/         # perplexity, generation, layerwise stats
├── integrations/       # HF, GGUF, safetensors, tokenizer loaders
└── cli/                # Hydra entrypoint
configs/                # Hydra hierarchy
tests/                  # pytest suite
notebooks/              # multimodal CPT notebook
```

> [!TIP]
> See `AGENTS.md` for the full Taichi invariants, core rules, and development workflows.

## Testing

Pytest is configured in `pyproject.toml`. Marker `slow` skips tests that download Qwen3.5-0.8B weights.

```bash
# Full suite
uv run pytest -v

# Skip slow tests
uv run pytest -m "not slow" -v

# Focused areas
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

- **Layer-type selection is explicit.** No block is swapped implicitly based on neighbouring layers. Use `config.layer_types[layer_idx]`.
- **Reward branch is additive only.** It never replaces the main mixer. Attaches only when listed in `config.parallel_reward_layers`, or when `use_parallel_reward=True` and the layer is attention-only.
- **Reward state persistence is explicit.** `GatedRewardNet.forward` returns `recurrent_state`, `value_baseline`, and `conv_state`. The wrapper writes them into `DynamicCache` under reward-specific keys. Never clobber GDN-2 cache.
- **GDN-2 gate shape contract.** Per-channel weights live inside the einsum: `einsum("bd,bdk,bk->bk", omega_W, dS, e)`. Do not broadcast a `[B, D]` × `[B, K]` outer product upstream.
- **Chain rule is multiplicative.** Given `omega_w_eff = omega_w * write`, recover `d_write = d_omega_w_eff * omega_w`. Do not divide by `write.clamp_min(1e-12)`.
- **Low-precision numerics.** Upcast to `float32` before `.pow()`, `.sum()`, `.mean()` on `float16` / `bfloat16`.
- **HF config hierarchy.** Multimodal configs nest `text_config` and `vision_config`. Never read top-level `hidden_size` on `InfiniDopamineConfig`; go through `text_config`.
- **Qwen3.5 weight loading uses `strict=True`.** New parameters need translation rules in `_qwen35_weights.py` / `_text_qwen35_weights.py`.

## Quality gates

```bash
uv run ruff check .
uv run pyrefly check
uv run pytest -m "not slow" -v
```

No `.github/`, no `Makefile`, no in-tree CI. Local `uv` commands are the quality gate.

## References

- Gated DeltaNet-2: [arXiv:2605.22791](https://arxiv.org/abs/2605.22791)
- Taichi: [taichi-lang.org](https://www.taichi-lang.org/)
