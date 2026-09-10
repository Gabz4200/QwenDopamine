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

> **Note:** This is research code, not a production-ready model. By default it builds on top of [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) weights.

## Why QwenDopamine?

Standard Qwen-style transformers handle local context beautifully but incur quadratic cost when dealing with long-horizon memory. QwenDopamine explores three complementary approaches:

* **Dual-stream Infini-attention:** Blends Gated DeltaNet-2 (GDN-2) linear recurrent memory with Sliding Window Attention (SWA), gated per head by a learnable sigmoid projection.
* **Gated Reward Net:** Adds a parallel reward branch with persistent recurrent, value, and conv states. It acts additively and never replaces the main mixer.
* **Taichi-accelerated kernels:** Custom forward and backward kernels for GDN-2 and Reinforced-Delta, with an automatic fallback to pure PyTorch references if Taichi is unavailable.

The framework also bundles Hugging Face and GGUF weight interoperability, Hydra-driven training configs, and a multimodal continued pre-training (CPT) pipeline that streams 16 datasets.

---

## Quick Start

### Prerequisites
* Python `>=3.12, <3.14`
* [uv](https://docs.astral.sh/uv/) package manager

### Installation

Set up your environment with `uv`. Pick the extras that match your hardware:

```bash
# CPU development environment (recommended for local work)
uv sync --extra cpu --extra dev --extra hf

# GPU (CUDA 12.8)
uv sync --extra gpu --extra dev --extra hf

# Multimodal CPT notebook extras
uv sync --extra cpt --extra hf
```

### Running the Training CLI

You can launch training either via the script directly or through the installed entrypoint:

```bash
# Run with the default config
uv run src/qwendopamine/cli/train.py

# Or via the installed entrypoint
uv run qwendopamine
```

**Overriding Configs:**
We use [Hydra](https://hydra.cc/) for configuration management. Overriding settings from the command line is straightforward:

```bash
uv run src/qwendopamine/cli/train.py model=infinidopamine_reference
uv run src/qwendopamine/cli/train.py train=single_gpu model=qwen35_gdn2
uv run src/qwendopamine/cli/train.py experiment=ablation_unfreezing
```

---

## Python API

> **Note:** The `qwendopamine` package uses [lazy loading](https://peps.python.org/pep-0562/) (PEP 562) to keep imports fast. Submodules like `qwendopamine.models` are only loaded when you first access an attribute from them, so `import qwendopamine` stays cheap.

### Loading a Model

```python
from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineForCausalLM,
)

config = InfiniDopamineConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
model = InfiniDopamineForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", config=config)
model.eval()
```

### Dynamic Model Factory

You can instantiate models dynamically using one of the built-in families (`qwen35`, `infinidopamine`, `research`):

```python
from qwendopamine.models.model_factory import create_model, build_model

model = create_model("infinidopamine", config)

# Or let the factory resolve the family directly from the config
model = build_model(config)
```

Want to add your own? Register it with the timm-style API:

```python
from qwendopamine.models.model_factory import register_model_family

register_model_family("my_model", my_builder_fn)
```

### Hugging Face & GGUF Interoperability

```python
# Save and reload through standard transformers APIs
model.save_pretrained("./ckpt")
reloaded = InfiniDopamineForCausalLM.from_pretrained("./ckpt")

# GGUF export / import
from qwendopamine.integrations.gguf import save_as_gguf, load_from_gguf
```

---

## Architecture Overview

The codebase separates model definitions, hardware-accelerated kernels, and training logic:

```text
src/qwendopamine/
├── models/               # Model implementations and configs
│   ├── infinidopamine/   # InfiniDopamine model & HF weight translations
│   ├── qwen35/           # Qwen3.5 baseline
│   ├── gdn2/             # GDN-2 blocks and recurrence logic
│   ├── reinforced/       # GatedRewardNet / Reinforced-Delta references
│   ├── blocks/           # BlockRegistry for dynamic layer building
│   └── core/             # Shared embeddings, RMSNorm, LM head
├── ops/                  # Public operations layer (handles backend dispatch)
├── kernels/taichi/       # Taichi kernels (forward + backward)
├── training/             # Training loops, schedules, and metrics
├── evaluation/           # Perplexity, generation, and layer-wise stats
├── integrations/         # HF, GGUF, and safetensors loaders
└── cli/                  # Hydra CLI entrypoints
configs/                  # Hydra configuration hierarchy
tests/                    # Pytest test suite
notebooks/                # Multimodal CPT Jupyter notebooks
```

When writing custom operations, use the public `qwendopamine.ops` layer rather than importing directly from the Taichi kernels. The ops layer routes calls to the Taichi backend automatically, or falls back to the pure PyTorch reference if Taichi isn't available.

---

## Configuration

Training configurations are managed via Hydra and live in the `configs/` directory. You can compose presets for models, training hardware, datasets, and experiments:

```bash
# Example: Qwen3.5 GDN-2 model, CPU training, debug dataset
uv run src/qwendopamine/cli/train.py model=qwen35_gdn2 train=cpu data=debug
```

See [`AGENTS.md`](AGENTS.md) for the full configuration directory layout and available preset names.

---

## Multimodal Continued Pre-training (CPT)

The `notebooks/train-infini-dopamine.ipynb` notebook (paired with a `.py` script via Jupytext) provides a complete CPT pipeline. It streams 16 Hugging Face datasets and pushes merged checkpoints to the Hub.

```bash
uv sync --extra cpt --extra hf
# Open notebooks/train-infini-dopamine.ipynb to get started
```

> **Heads up:** This notebook downloads large datasets and pushes checkpoints to the Hugging Face Hub. Make sure you have adequate disk space and network bandwidth before running it. If you edit the notebook, keep the `.py` pair in sync with `jupytext --sync`.

*(Tip: If running on Kaggle, use `accelerate.PartialState()` rather than `accelerate launch` or `torchrun`.)*

---

## Testing

We use `pytest` for the test suite. Tests that download `Qwen3.5-0.8B` weights are marked as `slow`.

```bash
# Run the full suite
uv run pytest -v

# Skip slow tests (recommended for quick local iteration)
uv run pytest -m "not slow" -v

# Run specific test areas
uv run pytest tests/models/test_gdn2.py -v
uv run pytest tests/ops/ -v
```

You can check the resolved Taichi backend with:

```bash
uv run python -c "from qwendopamine.kernels.taichi import taichi_arch; print(taichi_arch())"
```

If it returns `unavailable`, Taichi is not available and the pure PyTorch reference is used instead.

---

## Contributing

For development workflows, code quality setup, and the architectural invariants that keep this project training reliably, please see [`AGENTS.md`](AGENTS.md). It covers everything from adding new blocks and kernels to setting up your local environment, running linters, and the project's conventions.

---

## References

* **Gated DeltaNet-2:** [arXiv:2605.22791](https://arxiv.org/abs/2605.22791)
* **Taichi:** [taichi-lang.org](https://taichi-lang.org/)
* **Qwen3.5:** [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) on Hugging Face

