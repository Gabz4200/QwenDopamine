# QwenDopamine

QwenDopamine is a PyTorch research framework for Qwen-style LLM architectures, designed to evaluate novel recurrent memory blocks, altered state dynamics, and agentic performance enhancements.

It provides a pure PyTorch training and evaluation harness paired with Hugging Face `transformers` and GGUF weight interoperability.

## Key Features

- **InfiniDopamine Architecture**: Hybrid long-context architecture integrating dual-stream Infini-attention (SWA + GDN-2), pre-attention Gated Reward Networks, and Sliding Window Attention.
- **Dual-Stream Infini-Attention with GDN-2**: Combines local Sliding Window Attention (SWA) and Gated DeltaNet-2 ([arXiv:2605.22791](https://arxiv.org/abs/2605.22791)) via a learnable per-head gate $\beta$ ([arXiv:2404.07143](https://arxiv.org/abs/2404.07143)) over shared QKV projections.
- **Pre-Attention Gated Reward Networks**: Automatically routes linear layers immediately preceding attention to `GatedRewardNet`, providing advantage-guided memory updates and FiLM query modulation.
- **Continued Pre-Training from Qwen3.5**: Full parameter compatibility with pretrained Qwen3.5 checkpoints via automatic state dict pre-hooks with `strict=True` support.
- **CPU-First Local Environment**: Local development resolves against CPU PyTorch wheels via `uv`, isolating GPU dependencies (`flash-linear-attention`, `unsloth`) to remote/Kaggle environments to avoid Triton driver conflicts.
- **Strict Quality Gates**: Complete behavioral test suite with `pyrefly` type checking and `ruff` linting.

## InfiniDopamine Architecture

`InfiniDopamine` implements a three-tier sequence mixing strategy:

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │                   InfiniDopamine Layer Stack                │
                       └─────────────────────────────────────────────────────────────┘
                                                      │
               ┌──────────────────────────────────────┴──────────────────────────────────────┐
               ▼                                      ▼                                      ▼
    ┌──────────────────────┐               ┌──────────────────────┐               ┌──────────────────────┐
    │     GDN-2 Layer      │               │   GatedRewardNet     │               │     SWA Attention    │
    │ (Infini-Attention)   │               │ (Pre-Attention Layer)│               │   (Standalone SWA)   │
    ├──────────────────────┤               ├──────────────────────┤               ├──────────────────────┤
    │ Shared QKV Proj      │               │ Reinforced Delta     │               │ Standard QKV Proj    │
    │  ├─ SWA Stream       │               │  ├─ Memory Core      │               │ Rotary Pos Embed     │
    │  └─ GDN-2 Stream     │               │  ├─ Advantage Gate   │               │ Local Window Mask    │
    │ Per-Head Gate β      │               │  ├─ Value Baseline   │               │ Softmax Attention    │
    │ Gated RMSNorm + Out  │               │  └─ FiLM Modulator   │               │ Out Projection       │
    └──────────────────────┘               └──────────────────────┘               └──────────────────────┘
```

### 1. Dual-Stream Infini-Attention (`InfiniDopamineGatedDeltaNet`)

For standard linear recurrent layers, `InfiniDopamineGatedDeltaNet` computes two attention representations using the same $Q, K, V$ projections:

1. **GDN-2 Linear Recurrent Memory ($A_{\text{gdn2}}$)**: Decoupled channel-wise erase gate $\mathbf{b} \in \mathbb{R}^{H_v \times d_k}$ and write gate $\mathbf{w} \in \mathbb{R}^{H_v \times d_v}$ with chunkwise/recurrent delta updates.
2. **Local Sliding Window Attention ($A_{\text{swa}}$)**: Scaled dot-product attention restricted to a causal sliding window ($W=1024$ by default).

Each head combines the streams with an independent learnable parameter $\beta \in \mathbb{R}^{1 \times 1 \times H_v \times 1}$ (`betas`):

$$A = \text{sigmoid}(\beta) \odot A_{\text{swa}} + (1 - \text{sigmoid}(\beta)) \odot A_{\text{gdn2}}$$

### 2. Pre-Attention Gated Reward Net (`InfiniDopamineGatedRewardNet`)

Any linear recurrent layer immediately preceding an attention layer (`... -> GDN-2 -> GatedRewardNet -> Attention -> ...`) is instantiated as `InfiniDopamineGatedRewardNet`:

- Evaluates baseline expectations via an EMA value tracker.
- Computes advantage-guided update weights $\omega_t = 1.0 + \text{softsign}(W_{\text{adv}} [x_t, r_t, v_t])$.
- Conditions queries through FiLM modulation: $q_t \leftarrow \gamma(r_t) \odot q_t + \beta(r_t)$.
- Supports explicit `reward_values` injection during RL/fine-tuning while defaulting to neutral baseline behavior during pre-training.

### 3. Qwen3.5 Weight Compatibility

Pretrained Qwen3.5 weights load directly into `InfiniDopamineForCausalLM` with `strict=True`:
- Scalar erase gates expand across channel dimensions $d_k$ and $d_v$.
- Linear projections map into both GDN-2 and GatedRewardNet memory cores.
- Gating parameters ($\beta$) and reward conditioning weights initialize to standard neutral defaults ($\text{sigmoid}(\beta)=0.5$, $\omega_t=1.0$).

## Repository Structure

- `configs/` — Hydra configuration hierarchy for models (`infinidopamine_reference.yaml`, `qwen35_reference.yaml`), training protocols, and ablation studies.
- `src/qwendopamine/` — core Python package:
  - `models/infinidopamine/` — modular InfiniDopamine architecture, decoder layers, and causal LM heads.
  - `models/qwen35/` — modular Qwen3.5 baseline and weight mappings.
  - `models/gdn2/` — pure PyTorch reference kernels (`torch_chunk_gdn2`, `torch_recurrent_gdn2`) and `GatedRewardNet`.
  - `models/blocks/` — block registry (`BLOCKS`) and reward components.
  - `integrations/` — Hugging Face, GGUF, safetensors, and tokenizer loaders.
  - `training/` — training loop, learning rate schedules, freezing logic, and metrics.
  - `evaluation/` — perplexity computation, generation, and layerwise stats.
  - `cli/` — Hydra CLI entrypoints (`train.py`).
- `tests/` — behavioral pytest suite covering InfiniDopamine, GDN-2, reward components, and Qwen3.5.

## Setup & Usage

### Local CPU Environment

```bash
# Sync local CPU environment with dev & HF dependencies
uv sync --extra cpu --extra dev --extra hf

# Run Hydra training CLI
uv run src/qwendopamine/cli/train.py

# Run quality gates & tests
uv run pyrefly check
uv run ruff check .
uv run pytest -v
```

### Python API Example

```python
import torch
from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineForCausalLM,
    InfiniDopamineTextConfig,
)
from qwendopamine.models.qwen35 import Qwen3_5ForCausalLM

# 1. Instantiate InfiniDopamine with Sliding Window Attention & GDN-2
config = InfiniDopamineTextConfig(
    hidden_size=2048,
    num_hidden_layers=24,
    sliding_window=1024,
)
model = InfiniDopamineForCausalLM(config)

# 2. Transfer pretrained Qwen3.5 weights for continued pre-training
qwen_model = Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")
model.load_qwen35_weights(qwen_model, strict=True)

# 3. Forward pass supporting optional reward values
input_ids = torch.tensor([[10, 20, 30, 40]])
output = model(input_ids=input_ids)
```

## Development Conventions

- Use standard Hugging Face `transformers` for tokenization, generation, KV caching (`DynamicCache`), and checkpoint publishing.
- Use pure PyTorch for novel research blocks, altered residual connections, state dynamics, custom losses, and training loops.
- Keep `logs/`, `data/`, `checkpoints/`, `.venv/`, and `.aislop/` out of Git history.
