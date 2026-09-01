# QwenDopamine

QwenDopamine is a PyTorch research framework for Qwen-style LLM architectures, designed to evaluate novel recurrent memory blocks, altered state dynamics, and agentic performance enhancements.

It provides a pure PyTorch training and evaluation harness paired with Hugging Face `transformers` and GGUF weight interoperability.

## Key Features

- **InfiniDopamine Architecture**: Hybrid long-context architecture integrating dual-stream Infini-attention (SWA + GDN-2), parallel Gated Reward Networks, and Sliding Window Attention.
- **Dual-Stream Infini-Attention with GDN-2**: Combines local Sliding Window Attention (SWA) and Gated DeltaNet-2 ([arXiv:2605.22791](https://arxiv.org/abs/2605.22791)) via a learnable per-head gate $\beta$ ([arXiv:2404.07143](https://arxiv.org/abs/2404.07143)) over shared QKV projections.
- **Parallel Gated Reward Networks**: A `GatedRewardNet` fast-weight branch can be added in parallel to the main mixer with a data-dependent sigmoid gate initialized near zero (`sigmoid(-5) ≈ 0.0067`), preserving the pretrained main pathway while injecting reward-modulated plasticity.
- **Multimodal Vision Tower**: Inherits the Qwen3.5-VL vision encoder (`InfiniDopamineVisionModel`) for image+text CPT. Vision weights load directly from Qwen3.5 checkpoints via `InfiniDopamineForConditionalGeneration.load_qwen35_weights`.
- **Continued Pre-Training from Qwen3.5**: Full parameter compatibility with pretrained Qwen3.5 checkpoints via automatic state dict pre-hooks with `strict=True` support.
- **Dataset Mixer**: Streaming multi-dataset CPT with per-dataset schema formatters for 16+ HF datasets (SMB frames, maze traces, sokoban CoT, chess PGN, ALFWorld trajectories, etc.).
- **CPU-First Local Environment**: Local development resolves against CPU PyTorch wheels via `uv`, isolating GPU dependencies (`flash-linear-attention`, `unsloth`) to remote/Kaggle environments to avoid Triton driver conflicts.
- **Strict Quality Gates**: Complete behavioral test suite with `pyrefly` type checking and `ruff` linting.

## InfiniDopamine Architecture

`InfiniDopamine` is a tiered sequence mixing strategy. The main mixer of every
layer is selected explicitly by `config.layer_types[layer_idx]`; no implicit
replacement happens based on neighbouring layers. An optional reward branch
can run in parallel to that main mixer:

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │                   InfiniDopamine Decoder Layer              │
                       └─────────────────────────────────────────────────────────────┘
                                                      │
            ┌─────────────────────────────────────────┴──────────────────────────────────────────┐
            ▼                                                                                    ▼
   ┌────────────────────┐                                                              ┌────────────────────┐
   │     Main Mixer     │                                                              │   Parallel Branch  │
   │ (block_type based) │                                                              │ (opt-in reward)    │
   ├────────────────────┤                                                              ├────────────────────┤
   │ linear_attention / │                                                              │ InfiniDopamine     │
   │ gdn2 → GDN-2+SWA   │                                                              │ GatedRewardNet     │
   │ full/sliding → SWA │                                                              │  ├─ Memory Core    │
   │ gated_reward_net → │                                                              │  ├─ Advantage Gate │
   │ GatedRewardNet     │                                                              │  ├─ Value Baseline │
   └─────────┬──────────┘                                                              │  └─ FiLM Modulator │
             │                                                                         └─────────┬──────────┘
             ▼                                                                                   ▼
   shared input RMSNorm                                                       RMSNorm → sigmoid(W x + b) gate
             │                                                                  (init near 0, learned)
             │                                                                         │
             └───────────────────────  +  gate * reward_out  ───────────────────────────┘
```

### 1. Dual-Stream Infini-Attention (`InfiniDopamineGatedDeltaNet`)

For standard linear recurrent layers, `InfiniDopamineGatedDeltaNet` computes two attention representations using the same $Q, K, V$ projections:

1. **GDN-2 Linear Recurrent Memory ($A_{\text{gdn2}}$)**: Decoupled channel-wise erase gate $\mathbf{b} \in \mathbb{R}^{H_v \times d_k}$ and write gate $\mathbf{w} \in \mathbb{R}^{H_v \times d_v}$ with chunkwise/recurrent delta updates.
2. **Local Sliding Window Attention ($A_{\text{swa}}$)**: Scaled dot-product attention restricted to a causal sliding window ($W=1024$ by default).

Each head dynamically combines the streams using a data-dependent gating projection over input token states $x_t$ anchored by a learnable bias $\beta \in \mathbb{R}^{1 \times 1 \times H_v \times 1}$ (`betas`):

$$\text{gate}_{t} = \text{sigmoid}(\beta + W_{\text{gate}} x_t)$$
$$A_t = \text{gate}_t \odot A_{\text{swa}, t} + (1 - \text{gate}_t) \odot A_{\text{gdn2}, t}$$

Initialized with $W_{\text{gate}} = 0$ and $\beta = 0$, training begins at an exact 50/50 balance ($\text{gate}_t = 0.5$) regularized toward balance early on before dynamically routing per token as representations mature.

### 2. Parallel Gated Reward Net (`InfiniDopamineGatedRewardNet`)

The reward branch is **never** implicitly swapped in for the main mixer. It is
attached as a parallel branch to a layer only when the layer is explicitly
selected via `config.parallel_reward_layers` or when
`config.use_parallel_reward=True` and the layer is an attention-only layer
(`full_attention` / `sliding_attention`). The forward path is:

$$h_L = h_{L-1} + \text{RMSNorm}(f_{\text{main}}(\text{RMSNorm}(h_{L-1})))
        + \alpha \cdot \text{RMSNorm}(f_{\text{dopamine}}(\text{RMSNorm}(h_{L-1}), r))$$

with $\alpha = \sigma(W x + b)$, $W$ zero-initialized and $b$ defaulting to
$-5$ so $\sigma(b) \approx 0.0067$ at start. The main mixer is preserved
exactly; the dopamine branch is a small additive correction.

The branch itself contains:

- `RewardStatisticsExtractor` + `LearnableSoftsign` to turn raw reward signals
  into normalized statistics.
- A vectorial `ValueBaselineEMA` that tracks the running expectation of the
  reward vector.
- An `AdvantageGate` that collapses the advantage vector into a global
  modulation scalar $\omega_t \in (0, 2)$.
- A `DeltaMemoryCore` performing the gated delta-rule update
  $S_{t+1} = (1 - \omega E) \odot S_t + (\omega W) \odot (e_t k_t^T)$. The
  $\mathbf{d} \times \mathbf{d}$ state is optional and can be replaced by a
  low-rank `memory_rank=r` factorization (`U, V ∈ R^{B × d × r}`) to reduce
  memory and compute.
- FiLM-conditioned query readout: $q_t \leftarrow \gamma(r_t) \odot q_t + \beta(r_t)$.

State persistence is explicit: `GatedRewardNet` returns
`{"recurrent_state", "value_baseline", "conv_state"}` from every forward, and
`InfiniDopamineGatedRewardNet.forward` writes them into the Hugging Face
`DynamicCache` under reward-specific fields (`reward_recurrent_state`,
`reward_value_baseline`, `reward_conv_states`) so the GDN-2 branch state is
never clobbered.

### 3. Qwen3.5 Weight Compatibility

Pretrained Qwen3.5 weights load directly into `InfiniDopamineForConditionalGeneration` with `strict=True`:
- Scalar erase gates expand across channel dimensions $d_k$ and $d_v$.
- Linear projections map into both GDN-2 and GatedRewardNet memory cores.
- Vision tower weights (`model.visual.*`) load from Qwen3.5-VL checkpoint directly.
- Gating parameters ($\beta$) and reward conditioning weights initialize to standard neutral defaults ($\text{sigmoid}(\beta)=0.5$, $\omega_t=1.0$).

### 4. Multimodal Vision Tower

`InfiniDopamineForConditionalGeneration` inherits the Qwen3.5-VL vision encoder via `InfiniDopamineVisionConfig`:
- Vision encoder processes images → spatial feature maps → `merger.linear_fc1` + `merger.linear_fc2` project into language model hidden space.
- LoRA targets include `merger.linear_fc1` and `merger.linear_fc2` for efficient multimodal fine-tuning.
- `AutoProcessor` from `Qwen/Qwen3.5-0.8B` handles image preprocessing.

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
    InfiniDopamineForConditionalGeneration,
    InfiniDopamineTextConfig,
    InfiniDopamineVisionConfig,
)
from transformers import AutoProcessor, AutoTokenizer

# 1. Build multimodal config mirroring Qwen3.5-0.8B
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

# 2. Load pretrained Qwen3.5 weights (includes vision tower)
model.load_qwen35_weights("Qwen/Qwen3.5-0.8B", strict=True)

# 3. Prepare processor + tokenizer
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

# 4. Forward with optional reward_values for CPT
input_ids = torch.tensor([[10, 20, 30, 40]])
reward_values = torch.zeros_like(input_ids, dtype=torch.float32)
output = model(input_ids=input_ids, reward_values=reward_values)
```

## Multimodal Continued Pretraining

See `notebooks/train-infini-dopamine.ipynb` for the full multimodal CPT pipeline. That notebook:

- Streams **16 datasets** with per-dataset formatters
- Uses `interleave_datasets` for memory-efficient mixing
- Trains with LoRA on both text and vision tower projections
- Saves merged checkpoints and pushes to Hugging Face Hub

## Development Conventions

- Use standard Hugging Face `transformers` for tokenization, generation, KV caching (`DynamicCache`), and checkpoint publishing.
- Use pure PyTorch for novel research blocks, altered residual connections, state dynamics, custom losses, and training loops.
- Keep `logs/`, `data/`, `checkpoints/`, `.venv/`, and `.aislop/` out of Git history.
