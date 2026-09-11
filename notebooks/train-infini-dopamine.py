# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# --- Jupytext notebook ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
# --- End Jupytext ---

# %% [markdown.0]
# # InfiniDopamine Multimodal Continued Pretraining (CPT)
#
# Continued pretraining pipeline for `InfiniDopamineForConditionalGeneration` initialized from `Qwen/Qwen3.5-0.8B`.
# Streams and interleaves tokenized trajectory, reasoning, and world-model datasets with reward-conditioned forward passes.
#
# Runtime modes:
#   * **Kaggle** (`KAGGLE_KERNEL_RUN=true`, default on Kaggle): full run — 17 streaming datasets, real Qwen3.5-0.8B weights, GPU 4-bit AdamW.
#   * **Local** (any machine without `KAGGLE_KERNEL_RUN`): smoke-test run — synthetic offline dataset, tiny random-init text model, 2 steps (override with `QWD_LOCAL_STEPS`).

# %% [code.1]
# Runtime setup. Two environments are supported:
#
#   * Kaggle (KAGGLE_KERNEL_RUN=true): full continued-pretraining run — 17
#     streaming Hub datasets, real Qwen/Qwen3.5-0.8B weights transferred into
#     the InfiniDopamine architecture, 4-bit paged AdamW on GPU.
#   * Local (anything else): smoke-test run — a synthetic offline dataset, a
#     tiny random-init text-only model (fits a CPU/RAM-limited laptop), and
#     ``QWD_LOCAL_STEPS`` training steps (default 2).
#
# On Kaggle this cell installs the package from git with uv; transitive deps
# are constrained by pyproject bounds (numpy<2.5, scipy>=1.15) to avoid
# `ImportError: _center` via sklearn/scipy. The numpy/scipy/sklearn trio is
# force-reinstalled together so their compiled extensions stay ABI-matched.
# Locally it only verifies that the uv-provisioned environment is complete
# and never touches pip/uv (that would fight ``uv sync``).
import importlib.metadata
import importlib.util
import os
import subprocess
import sys

try:
    from packaging.version import Version
except ImportError:
    Version = None  # type: ignore[assignment]

IS_KAGGLE: bool = (
    os.path.isdir("/kaggle/working") or os.environ.get("KAGGLE_KERNEL_RUN") == "true"
)
LOCAL_TEST: bool = not IS_KAGGLE
_CAPPED_FULL: bool = os.environ.get("QWD_CAPPED_FULL_PIPELINE", "0") == "1"
_USE_SMOKE_CONFIG: bool = not IS_KAGGLE or _CAPPED_FULL
_MIN_TRANSFORMERS = Version("5.15.0") if Version is not None else None  # type: ignore[no-any-return]
_FORCE_INSTALL: bool = os.environ.get("QWD_DEBUG_INSTALL", "0") == "1"
_SKIP_INSTALL: bool = os.environ.get("QWD_SKIP_INSTALL", "0") == "1"
_SHOULD_INSTALL: bool = (IS_KAGGLE or _FORCE_INSTALL) and not _SKIP_INSTALL

if not _SHOULD_INSTALL:
    _REQUIRED_IMPORTS = {
        "qwendopamine": "qwendopamine",
        "accelerate": "accelerate",
        "datasets": "datasets",
        "peft": "peft",
        "trl": "trl",
        "transformers": "transformers",
    }
    _missing = [
        name
        for name, mod in _REQUIRED_IMPORTS.items()
        if importlib.util.find_spec(mod) is None
    ]
    if _missing:
        print(
            f"[setup] WARNING: local runtime missing {', '.join(sorted(_missing))}; "
            "provision with `uv sync --extra cpu --extra dev --extra hf --extra cpt` "
            "if running via uv."
        )
    try:
        _tf_ver = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        _tf_ver = "not installed"
    if (
        _tf_ver != "not installed"
        and Version is not None
        and _MIN_TRANSFORMERS is not None
        and Version(_tf_ver) < _MIN_TRANSFORMERS
    ):
        print(
            f"[setup] WARNING: transformers {_tf_ver} < {_MIN_TRANSFORMERS}; "
            "refresh the environment with `uv sync --extra cpt`."
        )
    print("[setup] Local runtime detected; skipping pip/uv (Kaggle-only).")
    _HF_TOKEN_FROM_SECRETS: str | None = None
else:
    _install_reason = "Kaggle" if IS_KAGGLE else "debug flag QWD_DEBUG_INSTALL"
    print(f"[setup] {_install_reason} detected — installing via uv...")
    _GIT_URL = "git+https://github.com/Gabz4200/QwenDopamine.git"
    # gpu: GPU torch via pytorch-cu128 index; cpt: streaming CPT deps (datasets/peft/trl/Pillow);
    # hf: datasets/tokenizers (overlaps cpt but kept for minimal Kaggle image).
    _PACKAGE_SPEC = os.environ.get(
        "QWD_PACKAGE_SPEC", f"qwendopamine[gpu,cpt,hf] @ {_GIT_URL}"
    )
    # Force-reinstall the numpy/scipy/sklearn trio together so their C
    # extensions stay ABI-matched. Without this a Kaggle image with a stale
    # scipy + new numpy yields `ImportError: _center` on the next cell.
    _COMPAT_PINS = [
        "numpy>=2.0,<2.5",
        "scipy>=1.15,<1.18",
        "scikit-learn>=1.6.0",
    ]

    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(cmd, check=False)
        except FileNotFoundError:
            return subprocess.CompletedProcess(cmd, returncode=127)

    def _pip_fallback(spec: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", spec],
            check=False,
        )

    # Step 1: compat trio
    _trio_cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--break-system-packages",
        "--upgrade",
        *_COMPAT_PINS,
    ]
    _proc = _run(_trio_cmd)
    if _proc.returncode != 0:
        for _pin in _COMPAT_PINS:
            _proc = _pip_fallback(_pin)
            if _proc.returncode != 0:
                break

    # Step 2: main package
    _uv_cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--break-system-packages",
        "--upgrade",
        _PACKAGE_SPEC,
    ]
    _proc = _run(_uv_cmd)
    if _proc.returncode != 0:
        _proc = _pip_fallback(_PACKAGE_SPEC)
    if _proc.returncode == 127:
        print("[setup] uv not found, falling back to pip...")
        _proc = _pip_fallback(_PACKAGE_SPEC)
    if _proc.returncode != 0:
        raise RuntimeError(
            "uv/pip install failed. On Kaggle, ensure Internet is ON and "
            "that the repo is reachable at https://github.com/Gabz4200/QwenDopamine."
        )
    # Validate trio in a fresh subprocess — the current interpreter may still
    # hold stale compiled extensions until the kernel restarts.
    _validate = subprocess.run(
        [sys.executable, "-c", "from scipy.sparse import csr_matrix; import sklearn; print('[setup] trio OK')"],
        check=False,
        capture_output=True,
        text=True,
    )
    if _validate.returncode != 0:
        print("[setup] WARNING: scipy/sklearn import failed after install:")
        print(_validate.stderr[-2000:])
        print("[setup] Restart the kernel and re-run (Kaggle: Kernel -> Restart).")
    else:
        print(_validate.stdout.strip())
    # Fetch HF token from Kaggle secrets (never from env directly).
    if IS_KAGGLE:
        try:
            from kaggle_secrets import (  # type: ignore[missing-import]
                UserSecretsClient,  # type: ignore[import-not-found]
            )
        except ImportError:
            _HF_TOKEN_FROM_SECRETS: str | None = None
        else:
            _kaggle_user_secrets = UserSecretsClient()
            _HF_TOKEN_FROM_SECRETS: str | None = _kaggle_user_secrets.get_secret(
                "HF_TOKEN"
            )

    else:
        _HF_TOKEN_FROM_SECRETS: str | None = None
    for _mod in list(sys.modules):
        if _mod == "PIL" or _mod.startswith("PIL."):
            sys.modules.pop(_mod, None)
    print("[setup] Done. Restart the kernel once and skip this cell on reruns.")

# %% [code.2]
import datetime
import gc
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F

# Fail fast if the Kaggle image has a stale scipy/numpy/sklearn trio.
# This must run before `peft`/`transformers` because transformers eagerly
# imports `sklearn.metrics` via `candidate_generator`, which pulls scipy and
# triggers `ImportError: _center` or `AttributeError: _blas_supports_fpe`
# when the trio is ABI-mismatched. The setup cell above reinstalls the trio,
# but the kernel must be restarted for the new compiled extensions to load.
try:
    import scipy.sparse
except (ImportError, AttributeError) as _trio_err:
    raise ImportError(
        "scipy import failed (likely numpy/scipy ABI mismatch after pip install). "
        "Restart the kernel (Kaggle: Kernel -> Restart) and skip the setup cell, "
        "then re-run. If using papermill, ensure the notebook install cell runs "
        "in a fresh kernel. Original error: " + str(_trio_err)
    ) from _trio_err
try:
    import scipy.sparse  # noqa: F401 re-check after sklearn pulled scipy
    import sklearn  # noqa: F401
except (ImportError, AttributeError) as _sk_err:
    raise ImportError(
        "sklearn/scipy import failed (likely numpy/scipy ABI mismatch after pip install). "
        "Restart the kernel (Kaggle: Kernel -> Restart) and skip the setup cell, "
        "then re-run. Original error: " + str(_sk_err)
    ) from _sk_err

from accelerate import PartialState
from datasets import IterableDataset, interleave_datasets, load_dataset

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        TrainingArguments,
    )
    from transformers.trainer_callback import TrainerCallback
    from trl import SFTTrainer
except (ImportError, AttributeError) as _imp_err:
    # Most common cause is the numpy/scipy _center / _blas_supports_fpe mismatch
    # bubbling through transformers -> sklearn -> scipy.
    if "_center" in str(_imp_err) or "numpy._core" in str(_imp_err) or "_blas_supports_fpe" in str(_imp_err):
        raise ImportError(
            "Import failed due to numpy/scipy ABI mismatch (`_center` / `_blas_supports_fpe` missing). "
            "Restart the kernel and re-run; the setup cell already reinstalled "
            "a compatible trio (numpy<2.5, scipy>=1.15, scikit-learn>=1.6). "
            "Original error: " + str(_imp_err)
        ) from _imp_err
    raise

from qwendopamine.integrations.huggingface import HFIntegration
from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineForCausalLM,
    InfiniDopamineForConditionalGeneration,
    InfiniDopamineTextConfig,
)

HFIntegration.register_infinidopamine_hf()

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

print("InfiniDopamine registered.")

# Rank/world size from accelerate.PartialState. Reads env vars set by Kaggle
# (T4 x2 -> world_size=2) or falls back to single-process for single T4 / CPU.
ACCEL_STATE = PartialState()
IS_MAIN = ACCEL_STATE.is_main_process
WORLD_SIZE = ACCEL_STATE.num_processes
RANK = ACCEL_STATE.process_index

if IS_MAIN:
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device         : {torch.cuda.get_device_name(i)}")
    print(f"World size     : {WORLD_SIZE}")

# Taichi backend: both mixers dispatch through qwendopamine.ops, which resolves
# to Taichi. The GDN-2 path (linear_attention mixer) needs the chunk Taichi op;
# the reward path (main mixer when layer_types selects it, plus the parallel
# branch) needs the delta Taichi op. Taichi works on any hardware, so we import
# and use it directly — a misconfigured env fails fast on its own.
from qwendopamine.kernels.taichi import taichi_arch
from qwendopamine.ops.reward import delta_core_step

_TAICHI_ARCH = taichi_arch()
if IS_MAIN:
    print(f"Taichi arch     : {_TAICHI_ARCH}")

# %% [markdown.3]
# ## Dataset Sources & Schema Mapping
#
# | Dataset | Source Format | Extracted Representation |
# |---|---|---|
# | `DylanRiden/smb-worldmodel-data` | Compressed `.npz` action arrays | Serialized 8-button action vectors |
# | `Kalso42/WorldModelForMaze` | Plaintext grid files (`.txt`) | Raw maze state tokens |
# | `ultrastar111/sokoban_...` | Serialized JSON `messages` | Dialogue / environment state trace |
# | `thuml/bytesized32-world-model-cot` | `prompt` + `reward_model` + `extra_info` | Structured CoT with reward targets |
# | `PatronusAI/world_model_corpus` | Chat messages with tool invocations | Formatted dialogue turns |
# | `schema-harness/arc-agi-3-schema-traces` | Benchmark score & level metadata | Serialized task run summaries |
# | `laion/strategic_game_chess` | SAN move sequence & outcome | PGN transcript |
# | `ryanmarten/OpenThoughts-1k-sample` | System + user/assistant turns | Conversation text |
# | `Decix/ReBel-ALFWorld-SFT-Trajectories` | Step observations & actions JSON | Action-observation trajectory |
# | `greghavens/kimi-k3-coding-and-debugging-traces` | Reasoning & code messages | Multi-turn reasoning traces |
# | `cot-leaderboard/cot-eval-traces-2.0` | Problem context, options, reasoning | Contextual CoT evaluation prompts |
# | `Lichess/standard-chess-games` | PGN game records with metadata | Annotated game moves |
# | `lockon/ToolACE` | Tool call conversations | Tool-use execution traces |
# | `faunix/Qwen3.8-27B-Distillation-40K` | Reasoning messages with domain tags | Tagged reasoning traces |
# | `Glint-Research/Fable-5-traces` | Agent execution traces | Agent trajectory transcripts |
# | `Salesforce/wikitext` | Raw text documents | Language modeling pretraining text |
# | `r0b0tlab/...-distillation` | JSON message logs with metadata | Multi-turn distillation traces |

# %% [code.4]
BASE_MODEL_NAME: str = "Qwen/Qwen3.5-0.8B"

CPT_DATASETS: list[str] = [
    "DylanRiden/smb-worldmodel-data",
    "Kalso42/WorldModelForMaze",
    "ultrastar111/sokoban_easy_v8_cot_chunk_kinf_world_model_20260707_perseg",
    "thuml/bytesized32-world-model-cot",
    "PatronusAI/world_model_corpus",
    "schema-harness/arc-agi-3-schema-traces",
    "laion/strategic_game_chess",
    "ryanmarten/OpenThoughts-1k-sample",
    "Decix/ReBel-ALFWorld-SFT-Trajectories",
    "greghavens/kimi-k3-coding-and-debugging-traces",
    "cot-leaderboard/cot-eval-traces-2.0",
    "Lichess/standard-chess-games",
    "lockon/ToolACE",
    "faunix/Qwen3.8-27B-Distillation-40K",
    "Glint-Research/Fable-5-traces",
    "Salesforce/wikitext",
    "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation",
    # ARC AGI 3 target — extremely important, spread across curriculum stages
    "AgentNativeResearchLab/arc-agi3-codex-gpt5.5-s5i5",
    "AgentNativeResearchLab/arc-agi3-kimi-k2.7-g50t",
    "AgentNativeResearchLab/arc-agi3-codex-gpt5.6sol-r11l",
    "AgentNativeResearchLab/arc-agi3-codex-gpt5.5-r11l",
    "nvidia/Nemotron-SFT-ARC-AGI-v1",
    "zhmz90/arc-agi-2",
    "dvilasuero/chain-of-draft-r1",
]

# ---------------------------------------------------------------------------
# Local smoke-test mode: bound the run size and use an offline synthetic
# dataset (see the dataset cell below).
# ---------------------------------------------------------------------------
LOCAL_SYNTHETIC_DATASET: str = "__local_synthetic__"
# Re-derive if kernel was restarted and setup cell skipped (setup defines these).
if "_USE_SMOKE_CONFIG" not in globals():
    import os as _os_fallback

    IS_KAGGLE = _os_fallback.path.isdir("/kaggle/working") or _os_fallback.environ.get("KAGGLE_KERNEL_RUN") == "true"  # type: ignore[no-redef]
    _CAPPED_FULL = _os_fallback.environ.get("QWD_CAPPED_FULL_PIPELINE", "0") == "1"  # type: ignore[no-redef]
    _USE_SMOKE_CONFIG = not IS_KAGGLE or _CAPPED_FULL  # type: ignore[no-redef]
if "_HF_TOKEN_FROM_SECRETS" not in globals():
    _HF_TOKEN_FROM_SECRETS = None  # type: ignore[no-redef]
if _USE_SMOKE_CONFIG and not _CAPPED_FULL:
    CPT_DATASETS = [LOCAL_SYNTHETIC_DATASET]

# Curriculum Learning: 5 stages easy→hard (Bengio 2009). Each stage
# holds ~4-5 datasets; after a stage finishes its cache/dataset objects
# are dropped (`del` + gc + cache wipe) before the next stage, so
# peak disk stays ~17GB max (Maze isolated) not 20GB+ for all 24.
# Order by transition-horizon: general language → ARC fundamentals →
# spatial world-models → ARC-AGI-3 agent trajectories (main target) →
# long-horizon world-reasoning. ARC datasets spread so early stages see
# ARC-2/CoD/schema, core ARC-3 agent stage sees 4 codex/kimi rollouts.
CURRICULUM_STAGES: dict[str, list[str]] = {
    # Stage 0 — Foundation language & distillation (short horizon, dense LM)
    "0_foundation": [
        "Salesforce/wikitext",
        "ryanmarten/OpenThoughts-1k-sample",
        "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation",
        "faunix/Qwen3.8-27B-Distillation-40K",
        "greghavens/kimi-k3-coding-and-debugging-traces",
    ],
    # Stage 1 — ARC fundamentals + efficient reasoning (ARC-2, CoD, schema, chess)
    # Introduces grid pattern + concise chain-of-draft before full ARC-3.
    "1_arc_foundation": [
        "zhmz90/arc-agi-2",
        "dvilasuero/chain-of-draft-r1",
        "schema-harness/arc-agi-3-schema-traces",
        "laion/strategic_game_chess",
        "Lichess/standard-chess-games",
    ],
    # Stage 2 — Spatial world-models + tool use (Maze isolated to this stage)
    "2_spatial": [
        "Kalso42/WorldModelForMaze",
        "ultrastar111/sokoban_easy_v8_cot_chunk_kinf_world_model_20260707_perseg",
        "lockon/ToolACE",
        "Decix/ReBel-ALFWorld-SFT-Trajectories",
        "thuml/bytesized32-world-model-cot",
    ],
    # Stage 3 — ARC-AGI-3 core agent trajectories (MAIN TARGET, 5 datasets)
    # 4 codex/kimi rollouts + Nemotron SFT (large_reasoning_and_tools).
    # All streaming, no snapshot, so 5 together is fine.
    "3_arc_agent": [
        "AgentNativeResearchLab/arc-agi3-codex-gpt5.5-s5i5",
        "AgentNativeResearchLab/arc-agi3-kimi-k2.7-g50t",
        "AgentNativeResearchLab/arc-agi3-codex-gpt5.6sol-r11l",
        "AgentNativeResearchLab/arc-agi3-codex-gpt5.5-r11l",
        "nvidia/Nemotron-SFT-ARC-AGI-v1",
    ],
    # Stage 4 — Reasoning world-models & long CoT (SMB isolated away from Maze)
    "4_reasoning_world": [
        "DylanRiden/smb-worldmodel-data",
        "PatronusAI/world_model_corpus",
        "cot-leaderboard/cot-eval-traces-2.0",
        "Glint-Research/Fable-5-traces",
    ],
}
# Start stage override for resuming/debugging (env QWD_CURRICULUM_STAGE).
CURRICULUM_START_STAGE: int = int(__import__("os").environ.get("QWD_CURRICULUM_STAGE", "0"))
USE_CURRICULUM: bool = not _USE_SMOKE_CONFIG and not _CAPPED_FULL
if USE_CURRICULUM:
    # Fail fast if any stage accidentally references synthetic/mocked data.
    _all_curriculum = [n for stage in CURRICULUM_STAGES.values() for n in stage]
    assert LOCAL_SYNTHETIC_DATASET not in _all_curriculum, "curriculum must use real datasets only"
    assert all(not n.startswith("__") for n in _all_curriculum), "no synthetic keys in curriculum"

DATASET_TEXT_COLUMN: str = "text"
MAX_SEQ_LENGTH: int = 1024

TORCH_DTYPE = (
    torch.float32
    if not torch.cuda.is_available()
    else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
)
# 4-bit paged AdamW is available on Kaggle via bitsandbytes, but kept off by default
# to avoid torchao/bnb conflict during smoke tests — enable for full CPT if needed.
LOAD_IN_4BIT: bool = False

USE_LORA: bool = True
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
USE_RSLORA: bool = True
# Targets resolved to nn.Linear; shared Qwen3.5 Linear weights are trained
# via QLoRA, exclusive InfiniDopamine Linear weights that can be LoRA are
# also via LoRA, and exclusive non-Linear weights are trained directly.
#
# Shared Linear (Qwen): q/k/v/o_proj, gate/up/down_proj — must be QLoRA.
# Exclusive Linear (GDN2 + reward): in_proj_*, out_proj, reward_* — LoRA.
# Exclusive non-Linear (conv, norm, A_log, dt_bias, betas, raw_alpha, gamma)
# are full-finetuned after PEFT via re_unfreeze_exclusive().
#
# trl>=0.24 defaults loss_type to chunked_nll, which is incompatible with a
# PEFT-wrapped lm_head. lm_head stays out of LoRA targets; it still trains
# via the EMBEDDING_LR_SCALE param group in CPTSFTTrainer.create_optimizer.
# embed_tokens is nn.Embedding, also trained directly.
LORA_TARGET_MODULES = [
    # GDN2 exclusive
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "in_proj_w",
    "in_proj_gate",
    "out_proj",
    # Shared attention
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # Shared mlp
    "gate_proj",
    "up_proj",
    "down_proj",
    # Reward branch (exclusive, Linear)
    "reward_gate_proj",
    "reward_branch.output_proj",
    "reward_branch.delta_layer.q_proj",
    "reward_branch.delta_layer.memory_core.k_proj",
    "reward_branch.delta_layer.memory_core.v_proj",
    "reward_branch.delta_layer.memory_core.w_proj",
    "reward_branch.delta_layer.memory_core.e_proj",
    "reward_branch.delta_layer.baseline_tracker.alpha_proj",
    "reward_branch.delta_layer.advantage_gate.plasticity_proj",
    "reward_branch.delta_layer.advantage_gate.write_proj",
    "reward_branch.delta_layer.advantage_gate.erase_proj",
    "reward_branch.delta_layer.reward_encoder.gamma_proj",
    "reward_branch.delta_layer.reward_encoder.beta_proj",
]

# Parallel GatedRewardNet branch configuration.
# Set USE_PARALLEL_REWARD=True to attach the dopamine branch to every
# attention-only layer. The branch is gated by sigmoid(W x + b) with
# REWARD_GATE_INIT_BIAS=-5 so sigmoid(b) ≈ 0.0067 at the start of training.
USE_PARALLEL_REWARD: bool = False
PARALLEL_REWARD_LAYERS: tuple[int, ...] = ()
REWARD_GATE_INIT_BIAS: float = -5.0
REWARD_MEMORY_RANK: int | None = None
PARALLEL_REWARD_GATE_LOSS_WEIGHT: float = 0.0
PARALLEL_REWARD_WARN_RATIO: float = 0.10
PARALLEL_REWARD_LOG_INTERVAL: int = 50

REWARD_LOSS_TYPE: str = "nll"  # "nll" | "ce" | "ppl"
REWARD_SCALE: float = 1.0
REWARD_EVERY_N_STEPS: int = 1
EMBEDDING_LR_SCALE: float = 0.2

# Reward-reference refresh: the per-token pseudo-reward pass is computed from a
# detached snapshot (a frozen copy of the current best model), not from the
# live training model. Refresh that snapshot at epoch boundaries and/or every
# ``REWARD_REFRESH_EVERY_N_STEPS`` optimizer steps (whichever fires first)
# after ``REWARD_REFRESH_WARMUP_STEPS`` steps have completed. When both are
# zero the snapshot is never refreshed (frozen at init).
REWARD_REFRESH_EVERY_N_EPOCHS: int = 0 if _USE_SMOKE_CONFIG else 1
REWARD_REFRESH_EVERY_N_STEPS: int = 0
REWARD_REFRESH_WARMUP_STEPS: int = 0 if _USE_SMOKE_CONFIG else 100

PER_DEVICE_TRAIN_BATCH_SIZE: int = 1
GRADIENT_ACCUMULATION_STEPS: int = 1 if _USE_SMOKE_CONFIG else 16
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 0.01
LR_SCHEDULER_TYPE: str = "cosine"
WARMUP_STEPS: int = 0 if _USE_SMOKE_CONFIG else 100
NUM_TRAIN_EPOCHS: int = 1
MAX_TRAIN_STEPS: int | None = (
    int(os.environ.get("QWD_LOCAL_STEPS", "2")) if _USE_SMOKE_CONFIG else None
)

LOGGING_STEPS: int = 1 if _USE_SMOKE_CONFIG else 10
SAVE_STEPS: int = 2 if _USE_SMOKE_CONFIG else 500
SAVE_TOTAL_LIMIT: int = 1 if _USE_SMOKE_CONFIG else 2

_RUN_ROOT = (
    os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")
    if not _USE_SMOKE_CONFIG
    else os.environ.get("QWD_LOCAL_RUN_DIR", os.path.join(os.getcwd(), "runs"))
)
# Include rank to avoid PID collision across distributed workers sharing a filesystem.
_RANK_SUFFIX = f"-rank{RANK}" if "RANK" in globals() else ""
OUTPUT_DIR: str = os.path.join(
    _RUN_ROOT,
    f"infini-dopamine-cpt-{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d-%H%M%S')}-{os.getpid()}{_RANK_SUFFIX}",
)
RESUME_FROM_CHECKPOINT: str | None = None
HUB_MODEL_ID: str = os.environ.get("HUB_MODEL_ID", "")
PUSH_TO_HUB: bool = False
HF_TOKEN: str | None = _HF_TOKEN_FROM_SECRETS or os.environ.get("HF_TOKEN", None)
MERGE_LORA_AFTER_TRAINING: bool = not _USE_SMOKE_CONFIG

SMB_CACHE_DIR: str = "./smb-cache"
MAZE_CACHE_DIR: str = "./maze-cache"

LICHESS_MAX_ROWS: int = 500_000
COT_EVAL_MAX_ROWS: int = 100_000
R0B0TLAB_MAX_ROWS: int = 200_000
WIKITEXT_MAX_ROWS: int = 50_000

# %% [markdown.5]
# ## Model Initialization & Weight Transfer
#
# Initializes `InfiniDopamineForConditionalGeneration` matching `Qwen/Qwen3.5-0.8B` architecture specifications and transfers overlapping base weights.
#
# The main mixer of every decoder layer is selected explicitly by
# `config.layer_types[layer_idx]`. The `GatedRewardNet` branch is no longer
# implicitly swapped in for layers that precede attention — it is attached
# as a **parallel** branch with a data-dependent sigmoid gate. Toggle
# `USE_PARALLEL_REWARD` to opt in for every attention-only layer, or pass
# an explicit `PARALLEL_REWARD_LAYERS` tuple to choose specific indices.


# %% [code.6]
def _build_infini_cfg(
    qwen_cfg: Any,
) -> InfiniDopamineConfig:
    """Build ``InfiniDopamineConfig`` from the upstream Qwen3.5 HF config."""
    hf_dict = qwen_cfg.to_dict()
    text_dict = dict(hf_dict.get("text_config", hf_dict))
    text_dict.setdefault("use_parallel_reward", USE_PARALLEL_REWARD)
    text_dict.setdefault("parallel_reward_layers", PARALLEL_REWARD_LAYERS)
    text_dict.setdefault("reward_gate_init_bias", REWARD_GATE_INIT_BIAS)
    text_dict.setdefault("reward_memory_rank", REWARD_MEMORY_RANK)
    text_dict.setdefault(
        "parallel_reward_gate_loss_weight", PARALLEL_REWARD_GATE_LOSS_WEIGHT
    )
    hf_dict["text_config"] = text_dict
    return InfiniDopamineConfig(**hf_dict)


tokenizer: Any = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if not _USE_SMOKE_CONFIG:
    # Only needed for multimodal (image/video) inputs on Kaggle; the text
    # training path never touches the processor.
    processor: Any = AutoProcessor.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=True
    )

HFIntegration.register_infinidopamine_hf()

if _USE_SMOKE_CONFIG:
    # --- Local smoke test: tiny random-init text-only model ----------------
    # Uses the cached Qwen3.5 tokenizer (offline). Vocab size falls back to
    # InfiniDopamineTextConfig's default (248320, matching Qwen3.5) so the
    # tokenizer never emits out-of-range ids. No base-weight transfer, no
    # vision tower, no network. Kept tiny so the run fits a CPU laptop.
    infini_cfg = InfiniDopamineTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention", "full_attention"],
    )
    model = InfiniDopamineForCausalLM(infini_cfg)
    missing: list[str] = []
    unexpected: list[str] = []
else:
    qwen_cfg = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    infini_cfg = _build_infini_cfg(qwen_cfg)
    model = InfiniDopamineForConditionalGeneration(infini_cfg)

    print("Loading base model weights...")
    _base_kwargs: dict[str, object] = {
        "dtype": TORCH_DTYPE,
        "device_map": "cpu",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig as _BNBConfig

        _base_kwargs["quantization_config"] = _BNBConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=TORCH_DTYPE,
        )
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **_base_kwargs)  # type: ignore[arg-type]

    # Each rank loads from the shared HF cache. Rank 0 pays the network fetch,
    # the rest hit disk. Avoids pickling ~1.6 GB of weights over the process group.
    missing, unexpected = model.load_qwen35_weights(base_model, strict=False)

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

ACCEL_STATE.wait_for_everyone()

print(f"Model type       : {model.config.model_type}")
print(f"Total params     : {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)
print(f"Missing keys     : {len(missing):,}")
print(f"Unexpected keys  : {len(unexpected):,}")
if missing:
    print("First missing keys:", missing[:10])
if unexpected:
    print("First unexpected keys:", unexpected[:10])
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} | {props.total_memory / 1024**3:.1f} GB")
        print(
            f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB | Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB"
        )


# %% [markdown.7]
# ## PEFT / LoRA Configuration


# %% [code.8]
def ensure_all_trainable(model: Any, missing_keys: list[str]) -> None:
    """Unfreeze newly initialized params so they are trainable."""
    missing_set = set(missing_keys)
    unfrozen = 0
    for name, param in model.named_parameters():
        if name in missing_set and not param.requires_grad:
            param.requires_grad = True
            unfrozen += 1
    if unfrozen:
        print(f"Unfrozen {unfrozen} newly initialized parameters.")


# Centralized substrings for direct-trained exclusive weights; keeps notebook and
# `qwendopamine.models` in sync — update here if model naming changes.
_NORM_SUBSTRINGS = frozenset(
    {"input_layernorm", "post_attention_layernorm", "q_norm", "k_norm", "model.norm"}
)
_GDN2_SUBSTRINGS = frozenset({"dt_bias", "A_log", "betas", "conv1d", ".norm"})
_REWARD_SUBSTRINGS = frozenset(
    {
        "scaler.raw_alpha",
        "stats_normalizer.gamma",
        "k_conv1d",
        "v_conv1d",
        "reward_branch_norm",
        "q_norm",
        "k_norm",
    }
)


def _should_unfreeze(name: str) -> bool:
    if "embed_tokens" in name or "lm_head" in name:
        return True
    if any(k in name for k in _NORM_SUBSTRINGS):
        return True
    if "linear_attn" in name and any(k in name for k in _GDN2_SUBSTRINGS):
        return True
    if "reward_branch" in name:
        if ".norm" in name:
            return True
        if any(k in name for k in _REWARD_SUBSTRINGS):
            return True
    return False


def re_unfreeze_reward_branch(model: Any) -> int:
    """Re-unfreeze direct-trained params PEFT froze.

    Handles exclusive InfiniDopamine non-Linear weights that cannot be LoRA:
    GDN2 (dt_bias, A_log, betas, conv1d, norm), reward branch (scaler,
    gamma, conv1d, norms), plus embed_tokens and lm_head which are trained
    via the embedding LR param group. Shared Linear weights stay frozen as
    base_layer with LoRA adapters (QLoRA) per the partition above.
    """
    unfrozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad or "lora" in name.lower():
            continue
        if _should_unfreeze(name):
            param.requires_grad = True
            unfrozen += 1
    if unfrozen:
        print(
            f"Re-unfrozen {unfrozen} direct-trained exclusive parameters after PEFT wrap."
        )
    return unfrozen


def re_unfreeze_exclusive(model: Any) -> int:
    """Alias for re_unfreeze_reward_branch covering all exclusive non-Linear."""
    return re_unfreeze_reward_branch(model)


lora_cfg = None
if USE_LORA:
    # PEFT LoRA only targets nn.Linear; embed_tokens is nn.Embedding.
    lora_targets = [m for m in LORA_TARGET_MODULES if m != "embed_tokens"]

    # Unfreeze fresh weights (e.g. parallel reward branch) BEFORE PEFT wraps.
    if missing:
        ensure_all_trainable(model, missing)

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=lora_targets,
        lora_dropout=LORA_DROPOUT,
        use_rslora=USE_RSLORA,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    # PEFT may have frozen non-Linear reward_branch params; re-unfreeze them.
    re_unfreeze_reward_branch(model)

    model.print_trainable_parameters()

# Final trainable parameter report.
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params     : {total_params:,}")
print(f"Trainable params : {trainable_params:,}")
print(f"Frozen params    : {total_params - trainable_params:,}")

model.train()
print("Model prepared for CPT.")

# %% [markdown.9]
# ## Dataset Formatting


# %% [code.10]
# Canonical formatter implementations also live in
# qwendopamine.integrations.cpt_datasets — keep in sync when editing.
def _flatten_messages(messages: Any) -> str:
    if isinstance(messages, str):
        stripped = messages.strip()
        if stripped.startswith(("[", "{")):
            try:
                messages = json.loads(stripped)
            except json.JSONDecodeError:
                return messages
        else:
            return messages
    if not isinstance(messages, list):
        return str(messages)
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("from", "user"))
            content = msg.get("content", msg.get("value", msg.get("text", "")))
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                parts.append(f"{role}: [thinking: {reasoning}]\n{content}")
            else:
                parts.append(f"{role}: {content}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def format_smb(example: dict) -> dict:
    return {"text": example.get("text", "")}


def format_maze(example: dict) -> dict:
    return {"text": example.get("text", "")}


def format_sokoban(example: dict) -> dict:
    messages_raw = example.get("messages", "")
    text = _flatten_messages(messages_raw)
    task = example.get("task", "")
    seed = example.get("seed", "")
    env_id = example.get("env_id", "")
    header = " | ".join(
        x for x in [f"task={task}", f"seed={seed}", f"env_id={env_id}"] if x
    )
    if header:
        text = f"[{header}]\n{text}"
    return {"text": text}


def format_bytesized32(example: dict) -> dict:
    prompt = example.get("prompt", [])
    reward_model = example.get("reward_model", "")
    extra_info = example.get("extra_info", "")
    parts = []
    if isinstance(prompt, list):
        for p in prompt:
            if isinstance(p, dict):
                role = p.get("role", "user")
                content = p.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(p))
    elif prompt:
        parts.append(str(prompt))
    if reward_model:
        parts.append(f"RewardModel: {reward_model}")
    if extra_info:
        parts.append(f"ExtraInfo: {extra_info}")
    return {"text": "\n".join(parts)}


def format_patronus(example: dict) -> dict:
    text = _flatten_messages(example.get("messages", []))
    return {"text": text}


def format_arc(example: dict) -> dict:
    task = example.get("task", "")
    status = example.get("status", "")
    win_levels = example.get("win_levels", "")
    level_scores = []
    for k in [f"level{i}" for i in range(10)]:
        if k in example and example[k] is not None:
            level_scores.append(f"{k}={example[k]}")
    scores_str = ", ".join(level_scores)
    text = (
        f"ARC-AGI Task [{task}] Status={status} WinLevels={win_levels}\n"
        f"Level Scores: {scores_str}"
    )
    return {"text": text}


def format_chess_laion(example: dict) -> dict:
    moves = example.get("Moves", [])
    termination = example.get("Termination", "")
    result = example.get("Result", "*")
    if isinstance(moves, list):
        movetext = " ".join(str(m) for m in moves)
    else:
        movetext = str(moves)
    text = f'[Event "?"]\n[Result "{result}"]\n\n{movetext} {termination}'
    return {"text": text}


def format_openthoughts(example: dict) -> dict:
    system = example.get("system", "")
    convs = example.get("conversations", [])
    parts = []
    if system:
        parts.append(f"system: {system}")
    for turn in convs:
        role = turn.get("from", turn.get("role", "unknown"))
        value = turn.get("value", turn.get("content", ""))
        parts.append(f"{role}: {value}")
    return {"text": "\n".join(parts)}


def format_alfworld(example: dict) -> dict:
    steps_raw = example.get("steps", "[]")
    steps: list[Any]
    if isinstance(steps_raw, str):
        stripped = steps_raw.strip()
        if stripped.startswith(("[", "{")):
            try:
                steps = json.loads(stripped)
            except json.JSONDecodeError:
                steps = []
        else:
            steps = []
    else:
        steps = steps_raw if isinstance(steps_raw, list) else []
    parts = []
    task = example.get("task", "")
    task_type = example.get("task_type", "")
    if task:
        parts.append(f"Task: {task}")
    if task_type:
        parts.append(f"Task Type: {task_type}")
    for step in steps:
        idx = step.get("idx", step.get("step", step.get("id", "")))
        obs = step.get("obs", step.get("observation", step.get("text", "")))
        action = step.get("action", step.get("act", ""))
        if obs:
            parts.append(f"Step {idx} Observation: {obs}")
        if action:
            parts.append(f"Step {idx} Action: {action}")
    return {"text": "\n".join(parts)}


def format_kimi_k3(example: dict) -> dict:
    text = _flatten_messages(example.get("messages", []))
    return {"text": text}


def format_cot_eval(example: dict) -> dict:
    parts = []
    passage = example.get("passage", "")
    if passage:
        parts.append(f"Passage: {passage}")
    question = example.get("question", "")
    if question:
        parts.append(f"Question: {question}")
    options = example.get("options", [])
    if options:
        opts = " | ".join(str(o) for o in options)
        parts.append(f"Options: {opts}")
    answer = example.get("answer", "")
    if answer:
        parts.append(f"Answer: {answer}")
    trace = example.get("reasoning_trace", "")
    if trace:
        parts.append(f"Reasoning: {trace}")
    return {"text": "\n".join(parts)}


def format_lichess(example: dict) -> dict:
    movetext = example.get("movetext", "")
    white = str(example.get("White") or "?")
    black = str(example.get("Black") or "?")
    result = example.get("Result", "*")
    opening = example.get("Opening", "")
    eco = example.get("ECO", "")
    event = example.get("Event", "")
    site = example.get("Site", "")
    date = example.get("UTCDate", "")
    text = (
        f'[Event "{event}"]\n[Site "{site}"]\n'
        f'[Date "{date}"]\n[White "{white}"]\n[Black "{black}"]\n'
        f'[Result "{result}"]\n[ECO "{eco}"]\n[Opening "{opening}"]\n\n{movetext}'
    )
    return {"text": text}


def format_toolace(example: dict) -> dict:
    system = example.get("system", "")
    convs = example.get("conversations", [])
    parts = []
    if system:
        parts.append(f"system: {system}")
    for turn in convs:
        role = turn.get("from", turn.get("role", "unknown"))
        value = turn.get("value", turn.get("content", ""))
        parts.append(f"{role}: {value}")
    return {"text": "\n".join(parts)}


def format_qwen3_distill(example: dict) -> dict:
    text = _flatten_messages(example.get("messages", []))
    domain = example.get("domain", "")
    category = example.get("category", "")
    source = example.get("source", "")
    meta = " | ".join(
        x for x in [f"domain={domain}", f"category={category}", f"source={source}"] if x
    )
    if meta:
        text = f"[{meta}]\n{text}"
    return {"text": text}


def format_fable5(example: dict) -> dict:
    text = _flatten_messages(example.get("messages", []))
    trace = example.get("trace", "")
    prompt = example.get("prompt", "")
    parts = []
    if prompt:
        parts.append(f"Prompt: {prompt}")
    parts.append(text)
    if trace:
        parts.append(f"Trace: {trace}")
    return {"text": "\n".join(parts)}


def format_wikitext(example: dict) -> dict:
    text = example.get("text", "")
    if not text or not text.strip():
        text = "[EMPTY_WIKITEXT_ROW]"
    return {"text": text}


def format_r0b0tlab(example: dict) -> dict:
    raw: list[Any] = []
    raw_value = example.get("messages_json", "[]")
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.startswith(("[", "{")):
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                raw = []
        else:
            raw = []
    else:
        raw = raw_value if isinstance(raw_value, list) else []
    text = _flatten_messages(raw)
    task_type = example.get("task_type", "")
    source = example.get("source", "")
    domain = example.get("domain", "")
    meta = " | ".join(
        x
        for x in [f"task_type={task_type}", f"source={source}", f"domain={domain}"]
        if x
    )
    if meta:
        text = f"[{meta}]\n{text}"
    return {"text": text}


def format_arc2(example: dict) -> dict:
    """zhmz90/arc-agi-2: train/test grids with filename."""
    filename = example.get("filename", "")
    parts = [f"ARC-AGI-2 [{filename}]"]
    train = example.get("train", [])
    test = example.get("test", [])
    if train:
        parts.append(f"Train examples: {len(train)}")
        for i, ex in enumerate(train[:3]):
            inp = ex.get("input", ex) if isinstance(ex, dict) else ex
            out = ex.get("output", "") if isinstance(ex, dict) else ""
            parts.append(f"  Train {i} input: {inp} -> output: {out}")
    if test:
        parts.append(f"Test examples: {len(test)}")
        for i, ex in enumerate(test[:2]):
            inp = ex.get("input", ex) if isinstance(ex, dict) else ex
            out = ex.get("output", "") if isinstance(ex, dict) else ""
            parts.append(f"  Test {i} input: {inp} -> output: {out}")
    return {"text": "\n".join(parts)}


def format_chain_of_draft(example: dict) -> dict:
    """dvilasuero/chain-of-draft-r1: question + CoD vs standard."""
    question = example.get("question", "")
    answer = example.get("answer", "")
    cod = example.get("cod", "")
    standard = example.get("standard", "")
    parts = []
    if question:
        parts.append(f"Question: {question}")
    # Prefer CoD (concise draft) for efficient reasoning; fall back to standard.
    reasoning = cod.strip() if cod and cod.strip() else standard
    if reasoning:
        parts.append(f"Reasoning: {reasoning}")
    if answer:
        parts.append(f"Answer: {answer}")
    return {"text": "\n".join(parts)}


def format_nemotron(example: dict) -> dict:
    """nvidia/Nemotron-SFT-ARC-AGI-v1: messages with system+user ARC puzzle."""
    text = _flatten_messages(example.get("messages", []))
    tools = example.get("tools", [])
    if tools:
        tool_names = ", ".join(t.get("name", "?") for t in tools if isinstance(t, dict))
        text = f"[Tools: {tool_names}]\n{text}"
    meta = example.get("metadata", "")
    if meta:
        text = f"{text}\n[metadata: {meta}]"
    return {"text": text}


def format_ara_agent(example: dict) -> dict:
    """AgentNativeResearchLab ARC-AGI-3 agent trajectories (episodes + ara stats).

    Streaming yields mixed rows: episodes (turn/action/state/frame) and
    accounting (ts/claims/trace_nodes). Handle both without dropping.
    """
    if "frame" in example:
        turn = example.get("turn", "")
        action = example.get("action", "")
        state = example.get("state", "")
        levels = example.get("levels_completed", "")
        frame = str(example.get("frame", ""))[:1200]
        return {"text": f"ARC3 Episode turn={turn} action={action} state={state} levels={levels}\nFrame:\n{frame}"}
    if "ts" in example:
        turn = example.get("turn", "")
        trace_nodes = example.get("trace_nodes", "")
        ara_bytes = example.get("ara_bytes", "")
        claims = example.get("claims", "")
        return {"text": f"ARC3 Trace ts={example.get('ts','')} turn={turn} nodes={trace_nodes} bytes={ara_bytes} claims={claims}"}
    # Fallback for ledger/predictions rows
    return {"text": " ".join(str(v)[:500] for v in example.values() if isinstance(v, (str, int, float)))}


DATASET_FORMATTERS = {
    "DylanRiden/smb-worldmodel-data": format_smb,
    "Kalso42/WorldModelForMaze": format_maze,
    "ultrastar111/sokoban_easy_v8_cot_chunk_kinf_world_model_20260707_perseg": format_sokoban,
    "thuml/bytesized32-world-model-cot": format_bytesized32,
    "PatronusAI/world_model_corpus": format_patronus,
    "schema-harness/arc-agi-3-schema-traces": format_arc,
    "laion/strategic_game_chess": format_chess_laion,
    "ryanmarten/OpenThoughts-1k-sample": format_openthoughts,
    "Decix/ReBel-ALFWorld-SFT-Trajectories": format_alfworld,
    "greghavens/kimi-k3-coding-and-debugging-traces": format_kimi_k3,
    "cot-leaderboard/cot-eval-traces-2.0": format_cot_eval,
    "Lichess/standard-chess-games": format_lichess,
    "lockon/ToolACE": format_toolace,
    "faunix/Qwen3.8-27B-Distillation-40K": format_qwen3_distill,
    "Glint-Research/Fable-5-traces": format_fable5,
    "Salesforce/wikitext": format_wikitext,
    "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation": format_r0b0tlab,
    "zhmz90/arc-agi-2": format_arc2,
    "dvilasuero/chain-of-draft-r1": format_chain_of_draft,
    "nvidia/Nemotron-SFT-ARC-AGI-v1": format_nemotron,
    "AgentNativeResearchLab/arc-agi3-codex-gpt5.5-s5i5": format_ara_agent,
    "AgentNativeResearchLab/arc-agi3-kimi-k2.7-g50t": format_ara_agent,
    "AgentNativeResearchLab/arc-agi3-codex-gpt5.6sol-r11l": format_ara_agent,
    "AgentNativeResearchLab/arc-agi3-codex-gpt5.5-r11l": format_ara_agent,
}


def format_example(example: dict, dataset_name: str) -> dict:
    formatter = DATASET_FORMATTERS.get(dataset_name)
    if formatter is not None:
        return formatter(example)
    for col in ["text", "content", "prompt", "problem", "solution"]:
        if example.get(col):
            return {"text": str(example[col])}
    text = " ".join(
        str(v)
        for v in example.values()
        if isinstance(v, (str, int, float)) and not str(v).startswith("_")
    )
    return {"text": text}


# %% [markdown.11]
# ## Streaming & Interleaving Pipeline


# %% [code.12]
def build_synthetic_dataset(
    num_docs: int = 512, words_per_doc: int = 48
) -> IterableDataset:
    """Offline smoke-test dataset: deterministic pseudo-random text rows.

    Only used in local test mode so the run never touches the Hub. The
    rows flow through the exact same ``map``/``filter``/collator/trainer
    pipeline as the Kaggle streaming datasets.
    """
    _words = [
        "synthetic",
        "reward",
        "memory",
        "delta",
        "state",
        "gate",
        "token",
        "world",
        "model",
        "smoke",
    ]

    def _gen() -> Iterator[dict]:
        for i in range(num_docs):
            text = " ".join(_words[(i + j) % len(_words)] for j in range(words_per_doc))
            yield {"text": f"doc {i}: {text}"}

    return IterableDataset.from_generator(_gen, gen_kwargs={})


def load_smb_dataset() -> IterableDataset:
    import io
    import zipfile

    from huggingface_hub import hf_hub_download

    repo_id = "DylanRiden/smb-worldmodel-data"
    cache_dir = Path(SMB_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename="smb_frames.zip",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )

    def _gen() -> Iterator[dict]:
        # Stream directly from the zip without extracting 118k files to disk.
        # The zip itself (~307MB) stays cached; we read one npz at a time via
        # BytesIO, so peak disk stays ~307MB instead of ~1GB+ extracted.
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = sorted(n for n in zf.namelist() if n.endswith(".npz"))
            print(f"Found {len(names)} SMB .npz files (streamed from {zip_path})")
            for name in names:
                with zf.open(name) as f:
                    data = np.load(io.BytesIO(f.read()))
                    action = data["action"]
                    if action.ndim == 0:
                        action = np.array([0.0] * 8)
                    buttons = ["Up", "Down", "Left", "Right", "A", "B", "Start", "Select"]
                    action_str = ", ".join(
                        f"{b}={float(v):.1f}" for b, v in zip(buttons, action.flatten()[:8])
                    )
                    yield {"text": f"SMB Frame Action: [{action_str}]"}

    return IterableDataset.from_generator(_gen, gen_kwargs={})


def load_maze_dataset() -> IterableDataset:
    from huggingface_hub import snapshot_download

    cache_dir = Path(MAZE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Deferred per-curriculum stage: this snapshot (~17GB, 1526 files) is only
    # downloaded when stage 2_spatial is active. Previous stages are streaming-only
    # (no snapshot), and _drop_stage_cache wipes this directory before stage 3,
    # so SMB (stage 3) and Maze never co-reside on disk.
    maze_dir = snapshot_download(
        repo_id="Kalso42/WorldModelForMaze",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    maze_path = Path(maze_dir)
    txt_files = sorted(maze_path.glob("data/**/*.txt"))
    if not txt_files:
        txt_files = sorted(maze_path.rglob("*.txt"))
    print(f"Found {len(txt_files)} maze .txt files")

    def _gen() -> Iterator[dict]:
        for tf in txt_files:
            text = tf.read_text(encoding="utf-8", errors="replace")
            text = text.strip()
            if text:
                yield {"text": text}

    return IterableDataset.from_generator(_gen, gen_kwargs={})


DATASET_CONFIG_MAP = {
    "PatronusAI/world_model_corpus": ("train", "train"),
    "Glint-Research/Fable-5-traces": ("pi_agent", "train"),
    "Salesforce/wikitext": ("wikitext-103-raw-v1", "train"),
    "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation": ("sft_balanced", "train"),
    "schema-harness/arc-agi-3-schema-traces": ("default", "test"),
    "nvidia/Nemotron-SFT-ARC-AGI-v1": ("large_reasoning_and_tools", "train"),
}

DATASET_SUBSET_MAP = {
    "Lichess/standard-chess-games": LICHESS_MAX_ROWS,
    "cot-leaderboard/cot-eval-traces-2.0": COT_EVAL_MAX_ROWS,
    "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation": R0B0TLAB_MAX_ROWS,
    "Salesforce/wikitext": WIKITEXT_MAX_ROWS,
}


def _capped_rows() -> int:
    return int(os.environ.get("QWD_CAPPED_ROWS", "5"))


def _mock_stream_for_dataset(name: str, n: int | None = None) -> IterableDataset:
    if n is None:
        n = _capped_rows()

    def _gen() -> "Iterator[dict]":
        for i in range(n):
            if name == "DylanRiden/smb-worldmodel-data":
                raw: dict = {"text": f"SMB Frame Action: [Up={i}.0, Down=0.0]"}
            elif name == "Kalso42/WorldModelForMaze":
                raw = {"text": f"maze {i}\n###\n# {i} #\n###"}
            elif (
                name
                == "ultrastar111/sokoban_easy_v8_cot_chunk_kinf_world_model_20260707_perseg"
            ):
                raw = {
                    "messages": json.dumps(
                        [{"role": "user", "content": f"sokoban {i}"}]
                    ),
                    "task": "t",
                    "seed": str(i),
                    "env_id": "e",
                }
            elif name == "thuml/bytesized32-world-model-cot":
                raw = {
                    "prompt": [{"role": "user", "content": f"bytesized {i}"}],
                    "reward_model": "rm",
                    "extra_info": "{}",
                }
            elif name == "PatronusAI/world_model_corpus":
                raw = {"messages": [{"role": "user", "content": f"patronus {i}"}]}
            elif name == "schema-harness/arc-agi-3-schema-traces":
                raw = {
                    "task": f"arc{i}",
                    "status": "ok",
                    "win_levels": "1",
                    **{f"level{j}": j for j in range(3)},
                }
            elif name == "laion/strategic_game_chess":
                raw = {"Moves": ["e4", "e5"], "Termination": "*", "Result": "1-0"}
            elif name == "ryanmarten/OpenThoughts-1k-sample":
                raw = {
                    "system": "sys",
                    "conversations": [
                        {"from": "human", "value": f"thought {i}"},
                        {"from": "gpt", "value": "ans"},
                    ],
                }
            elif name == "Decix/ReBel-ALFWorld-SFT-Trajectories":
                raw = {
                    "steps": json.dumps(
                        [{"idx": 0, "obs": f"obs {i}", "action": "act"}]
                    ),
                    "task": f"alf {i}",
                    "task_type": "t",
                }
            elif name == "greghavens/kimi-k3-coding-and-debugging-traces":
                raw = {"messages": [{"role": "user", "content": f"kimi {i}"}]}
            elif name == "cot-leaderboard/cot-eval-traces-2.0":
                raw = {
                    "passage": f"p {i}",
                    "question": "q?",
                    "options": ["a", "b"],
                    "answer": "a",
                    "reasoning_trace": "trace",
                }
            elif name == "Lichess/standard-chess-games":
                raw = {
                    "movetext": "1. e4 e5",
                    "White": "A",
                    "Black": "B",
                    "Result": "*",
                    "Opening": "o",
                    "ECO": "C20",
                    "Event": "ev",
                    "Site": "s",
                    "UTCDate": "2024.01.01",
                }
            elif name == "lockon/ToolACE":
                raw = {
                    "system": "sys",
                    "conversations": [{"from": "human", "value": f"tool {i}"}],
                }
            elif name == "faunix/Qwen3.8-27B-Distillation-40K":
                raw = {
                    "messages": [{"role": "user", "content": f"distill {i}"}],
                    "domain": "d",
                    "category": "c",
                    "source": "s",
                }
            elif name == "Glint-Research/Fable-5-traces":
                raw = {
                    "messages": [{"role": "user", "content": f"fable {i}"}],
                    "trace": "tr",
                    "prompt": "p",
                }
            elif name == "Salesforce/wikitext":
                raw = {"text": f"wikitext doc {i} with some language modeling text"}
            elif name == "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation":
                raw = {
                    "messages_json": json.dumps(
                        [{"role": "user", "content": f"r0b0t {i}"}]
                    ),
                    "task_type": "t",
                    "source": "s",
                    "domain": "d",
                }
            else:
                raw = {"text": f"fallback {name} {i}"}
            yield {"text": format_example(raw, name)["text"]}

    return IterableDataset.from_generator(_gen, gen_kwargs={})


def _apply_subset(ds: Any, dataset_name: str) -> IterableDataset:
    max_rows = DATASET_SUBSET_MAP.get(dataset_name)
    if max_rows is None:
        result: IterableDataset = ds
        return result

    def _gen() -> Iterator[dict]:
        for i, ex in enumerate(ds):
            yield ex
            if i >= max_rows:
                break

    return IterableDataset.from_generator(_gen, gen_kwargs={})


def _stream_for(name: str, use_capped: bool) -> IterableDataset:
    if name == LOCAL_SYNTHETIC_DATASET and IS_KAGGLE:
        raise RuntimeError("synthetic dataset not allowed on Kaggle; curriculum must use real datasets")
    if use_capped:
        return _mock_stream_for_dataset(name)
    if name == LOCAL_SYNTHETIC_DATASET:
        return build_synthetic_dataset()
    if name == "DylanRiden/smb-worldmodel-data":
        return load_smb_dataset()
    if name == "Kalso42/WorldModelForMaze":
        return load_maze_dataset()
    cfg, split = DATASET_CONFIG_MAP.get(name, ("default", "train"))
    ds = load_dataset(name, config=cfg, split=split, streaming=True)

    def _fmt(ex: dict, dataset_name: str = name) -> dict:
        return format_example(ex, dataset_name)

    fmt = DATASET_FORMATTERS.get(name)
    if fmt is not None:
        ds = ds.map(_fmt, batched=False)
    result: IterableDataset = _apply_subset(ds, name)
    return result


def build_streaming_dataset(
    dataset_names: list[str],
    seed: int = 42,
) -> IterableDataset:
    use_capped = _CAPPED_FULL
    if use_capped:
        print(
            f"[capped-full] using {_capped_rows()} mocked rows per dataset for {len(dataset_names)} datasets"
        )
    streams = [_stream_for(n, use_capped) for n in dataset_names]
    if len(streams) == 1:
        return streams[0]
    # all_exhausted ensures wikitext (50k capped) doesn't stop larger streams early — the
    # 17-way interleave keeps sampling until every source is drained, balancing world-model traces.
    return interleave_datasets(streams, seed=seed, stopping_strategy="all_exhausted")


def peek_streaming_dataset(dataset_names: list[str], seed: int = 42) -> IterableDataset:
    rank_seed = seed + RANK
    train_dataset = build_streaming_dataset(dataset_names, seed=rank_seed)
    if IS_MAIN:
        sample = next(iter(train_dataset.take(1)), None)
        if sample is None:
            print("[peek] warning: streaming dataset returned no samples")
        else:
            print(f"Sample keys  : {list(sample.keys())}")
            print(f"Sample text  : {str(sample.get('text', ''))[:240]}")
            print(f"Sample length: {len(str(sample.get('text', '')))}")
            del sample
            gc.collect()
    return train_dataset


def _drop_stage_cache(stage_datasets: list[str]) -> None:
    """Remove snapshot caches for datasets in the finished stage to free disk."""
    import shutil

    if "DylanRiden/smb-worldmodel-data" in stage_datasets:
        p = Path(SMB_CACHE_DIR)
        if p.exists():
            print(f"[curriculum] dropping SMB cache {p}")
            shutil.rmtree(p, ignore_errors=True)
    if "Kalso42/WorldModelForMaze" in stage_datasets:
        p = Path(MAZE_CACHE_DIR)
        if p.exists():
            print(f"[curriculum] dropping Maze cache {p}")
            shutil.rmtree(p, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if USE_CURRICULUM:
    _curriculum_keys = list(CURRICULUM_STAGES.keys())
    _active_keys = _curriculum_keys[CURRICULUM_START_STAGE:]
    print(f"[curriculum] {len(CURRICULUM_STAGES)} stages total; starting at {CURRICULUM_START_STAGE}: {', '.join(_active_keys)}")
    # Defer dataset build to stage loop in the training cell; keep placeholder.
    train_dataset = None  # type: ignore[assignment]
else:
    train_dataset = peek_streaming_dataset(CPT_DATASETS)


# %% [code.13]
def tokenize_fn(example: dict) -> dict:
    text = (example.get("text") or "").strip()
    if not text:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    tok = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)
    if not tok["input_ids"]:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "labels": tok["input_ids"],
    }


# Keep column logic as a function so curriculum stages can reuse it per-stage.
def _cols_to_remove_for(ds: Any) -> list[str]:
    return [c for c in (getattr(ds, "column_names", None) or []) if c not in {"text", "input_ids", "attention_mask", "labels"}]


def _keep_tokenized(example: dict) -> bool:
    return len(example.get("input_ids") or []) > 0


from transformers import DataCollatorWithPadding

# Collator pads to longest in batch (batch=1, so no-op) — packing is off and sequences are
# pre-truncated to MAX_SEQ_LENGTH, so DataCollatorForLanguageModeling would duplicate labels.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

if not USE_CURRICULUM:
    cols_to_remove = _cols_to_remove_for(train_dataset)
    train_dataset = train_dataset.map(  # type: ignore[union-attr]
        tokenize_fn,
        batched=False,
        remove_columns=cols_to_remove,
    )
    train_dataset = train_dataset.filter(_keep_tokenized)  # type: ignore[union-attr]
    print(f"Tokenized columns : {train_dataset.column_names}")  # type: ignore[union-attr]
    print(f"Max seq length    : {MAX_SEQ_LENGTH}")
else:
    # Tokenization deferred: each curriculum stage builds and tokenizes its own dataset
    # inside the training cell so disk from the previous stage can be freed first.
    print("[curriculum] tokenization deferred to stage loop")

# %% [markdown.14]
# ## Reward Conditioning, Parallel Branch & Custom Trainer
#
# Computes per-token pseudo-rewards via a detached base-model pass and passes
# `reward_values` into `InfiniDopamine` during training.
#
# The reward pass uses a **frozen snapshot** of the model (the "reward reference"
# model), not the live training model. This decouples the reward signal from
# gradient noise and prevents the reward from chasing the model's own updates
# within the same step.
#
# ## Reward Reference Refresh
#
# The snapshot is periodically refreshed from the current best checkpoint so the
# reward signal tracks the evolving model instead of staying frozen at init.
# Refresh happens at the earlier of:
#   * Epoch boundary, when `REWARD_REFRESH_EVERY_N_EPOCHS > 0`
#   * Every `REWARD_REFRESH_EVERY_N_STEPS` optimizer steps, when
#     `REWARD_REFRESH_EVERY_N_STEPS > 0`
#
# Both are gated by `REWARD_REFRESH_WARMUP_STEPS` — no refresh occurs until that
# many optimizer steps have completed. When both refresh intervals are `0` the
# snapshot stays frozen at initialisation (useful for short smoke tests).
#
# When `USE_PARALLEL_REWARD=True` (or `PARALLEL_REWARD_LAYERS` is non-empty)
# the trainer also exposes diagnostics for the parallel `GatedRewardNet`
# branch: gate mean/max, effective branch contribution, EMA value baseline,
# and fast-weight state norm. These are logged every
# `PARALLEL_REWARD_LOG_INTERVAL` steps so the dopamine branch stays
# observable in TensorBoard even before reward signals become meaningful.


# %% [code.15]
import re

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(run_dir: str | os.PathLike[str]) -> Path | None:
    """Return the highest-numbered ``checkpoint-*`` dir under *run_dir*, or None."""
    root = Path(run_dir)
    if not root.is_dir():
        return None
    best: Path | None = None
    best_step = -1
    for entry in root.iterdir():
        m = _CHECKPOINT_RE.match(entry.name)
        if m and entry.is_dir():
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best = entry
    return best


def build_reward_values(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_ref: Any,
) -> torch.Tensor:
    """Compute per-token pseudo-rewards from a detached base-model pass.

    Alignment: when generating x(t+1) from input x(t), we consume
    the reward of token x(t). This is achieved by shifting rewards
    one position forward: reward_values[:, 1:] = rewards.

    ``model_ref`` is set to eval mode (so dropout is disabled during
    reward estimation) and restored to its prior training state afterwards.
    """
    if model_ref is None:
        raise ValueError("model_ref must be provided")
    _model = model_ref
    _was_training = _model.training
    _model.eval()
    # Ref model may stay on CPU (deepcopy) while training model is on GPU.
    try:
        _ref_device = next(_model.parameters()).device
    except StopIteration:
        _ref_device = input_ids.device
    _input_ids = input_ids.to(_ref_device)
    _attention_mask = attention_mask.to(_ref_device)
    with torch.no_grad():
        base_outputs = _model(
            input_ids=_input_ids,
            attention_mask=_attention_mask,
        )
        shift_logits = base_outputs.logits[..., :-1, :].contiguous()
        shift_labels = _input_ids[..., 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(_input_ids.size(0), -1)

        if REWARD_LOSS_TYPE in ("nll", "ce"):
            rewards = -token_loss * REWARD_SCALE
        else:
            rewards = (-token_loss).exp() * REWARD_SCALE

        reward_values = torch.zeros_like(input_ids, dtype=TORCH_DTYPE)
        reward_values[:, 1:] = rewards.to(TORCH_DTYPE).to(input_ids.device)
        reward_values = reward_values * attention_mask

    if _was_training:
        _model.train()
    return reward_values


class CPTSFTTrainer(SFTTrainer):
    def create_optimizer(self, model: Any = None) -> Any:
        """Create optimizer with lower LR for embedding layers."""
        opt_model = self.model if model is None else model
        if self.optimizer is not None:
            return self.optimizer

        embed_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ("embed_tokens", "lm_head")):
                embed_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": self.args.learning_rate},
            {
                "params": embed_params,
                "lr": self.args.learning_rate * EMBEDDING_LR_SCALE,
            },
        ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def __init__(
        self,
        *args: Any,
        reward_every_n_steps: int = 1,
        reward_ref_refresh_epochs: int = 0,
        reward_ref_refresh_steps: int = 0,
        reward_ref_warmup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reward_every_n_steps = max(1, reward_every_n_steps)
        self._global_step = 0
        # Frozen snapshot used for pseudo-reward computation. When refresh
        # is enabled (>0 on either axis) it is periodically updated from the
        # current best checkpoint; otherwise it stays fixed at init.
        self._reward_ref_refresh_epochs = max(0, reward_ref_refresh_epochs)
        self._reward_ref_refresh_steps = max(0, reward_ref_refresh_steps)
        self._reward_ref_warmup_steps = max(0, reward_ref_warmup_steps)
        self._last_epoch_refreshed: float | None = None
        self._last_step_refreshed: int = 0
        self._reward_ref_model: Any = None
        self._init_reward_ref()
        # Register the reward-refresher callback so the snapshot updates
        # at epoch/step boundaries when refresh is enabled.
        self.add_callback(_rewards_refresher)
        _rewards_refresher.bind(self)

    def _init_reward_ref(self) -> None:
        """Create the frozen reward-reference model as a deep copy of the
        base (PEFT-stripped) model so the reward pass is detached from live
        gradients and adapter updates."""
        import copy

        if USE_LORA and hasattr(self.model, "base_model"):
            base_for_ref = self.model.base_model
        else:
            base_for_ref = self.model
        self._reward_ref_model = copy.deepcopy(base_for_ref)
        self._reward_ref_model.requires_grad_(False)
        self._reward_ref_model.eval()
        if IS_MAIN:
            print("Reward ref model initialised (frozen snapshot).")

    def refresh_reward_ref(self) -> bool:
        """Re-snapshot the reward reference model from the latest checkpoint.

        Merges pending LoRA adapter state (so the ref is a full-weight copy)
        then loads the most recent ``checkpoint-*`` directory under
        ``OUTPUT_DIR``. Returns ``True`` when a refresh was performed.
        No-op when refresh is disabled or no checkpoint is found.
        """
        if self._reward_ref_refresh_epochs == 0 and self._reward_ref_refresh_steps == 0:
            return False

        latest = find_latest_checkpoint(self.args.output_dir)
        if latest is None:
            if IS_MAIN:
                print("refresh_reward_ref: no checkpoint found yet, skipping.")
            return False

        if IS_MAIN:
            print(f"refresh_reward_ref: loading from {latest}")

        peft_model = self.model
        # Merge LoRA into the base model so we can copy a single weight set.
        if USE_LORA and hasattr(peft_model, "merge_and_unload"):
            merged = peft_model.merge_and_unload()  # type: ignore[not-callable]
            merged_state = merged.state_dict()
        else:
            merged_state = peft_model.state_dict()

        # Align devices: merged state may be on GPU while ref stays on CPU.
        try:
            _ref_device = next(self._reward_ref_model.parameters()).device
        except StopIteration:
            _ref_device = torch.device("cpu")
        merged_state = {
            k: v.to(_ref_device) if isinstance(v, torch.Tensor) else v
            for k, v in merged_state.items()
        }

        # Load into the ref model (strict=False so new init weights for
        # exclusive layers are preserved when they have no checkpoint entry).
        missing, unexpected = self._reward_ref_model.load_state_dict(
            merged_state, strict=False
        )
        self._reward_ref_model.eval()
        self._reward_ref_model.requires_grad_(False)
        if IS_MAIN:
            print(
                f"refresh_reward_ref done (missing={len(missing)} "
                f"unexpected={len(unexpected)})."
            )
        return True

    def compute_loss(
        self,
        model: Any,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.to(model.device)

        if self._global_step % self.reward_every_n_steps == 0:
            ref = (
                self._reward_ref_model if self._reward_ref_model is not None else model
            )
            reward_values = build_reward_values(input_ids, attention_mask, ref)
        else:
            reward_values = torch.zeros_like(
                input_ids, dtype=TORCH_DTYPE, device=model.device
            )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            reward_values=reward_values,
        )
        self._global_step += 1

        self._log_parallel_reward_metrics(model, outputs)

        loss = outputs.loss
        if return_outputs:
            result_tuple: tuple[torch.Tensor, Any] = (loss, outputs)
            return result_tuple
        result: torch.Tensor = loss
        return result

    def _log_parallel_reward_metrics(self, model, outputs) -> None:
        r"""Surface parallel reward branch diagnostics on a fixed cadence.

        Uses :func:`collect_parallel_reward_metrics` so the metrics format
        matches the one produced by :class:`TrainingLoop` and any other
        trainer in the package. Warnings are emitted to stdout when the
        branch starts contributing more than the configured fraction of the
        main path norm.
        """
        if not USE_PARALLEL_REWARD:
            return
        if self._global_step % max(1, PARALLEL_REWARD_LOG_INTERVAL) != 0:
            return
        from qwendopamine.training import (
            collect_parallel_reward_metrics,
            maybe_warn_branch_ratio,
        )

        cache = getattr(outputs, "past_key_values", None)
        metrics = collect_parallel_reward_metrics(
            model,
            past_key_values=cache,
        )
        if not metrics:
            return
        formatted = ", ".join(
            f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
            for name, value in metrics.items()
        )
        print(f"[parallel_reward step={self._global_step}] {formatted}")
        warning = maybe_warn_branch_ratio(metrics, PARALLEL_REWARD_WARN_RATIO)
        if warning is not None:
            print(f"[parallel_reward WARN] {warning}")


class RewardRefresher(TrainerCallback):
    """Callback that refreshes the trainer's reward-reference snapshot.

    Fires at epoch boundaries (if ``reward_ref_refresh_epochs > 0``) and/or
    every ``reward_ref_refresh_steps`` optimizer steps (if
    ``reward_ref_refresh_steps > 0``), skipping until
    ``reward_ref_warmup_steps`` have elapsed. Delegates to
    :meth:`CPTSFTTrainer.refresh_reward_ref`.

    The trainer back-reference is set by :class:`CPTSFTTrainer` after
    construction (the HF ``CallbackHandler`` does not inject it automatically).
    """

    def __init__(self) -> None:
        self._trainer: CPTSFTTrainer | None = None

    def bind(self, trainer: CPTSFTTrainer) -> None:
        self._trainer = trainer

    def _should_refresh_step(self, state: Any) -> bool:
        t = self._trainer
        if t is None:
            return False
        step_interval = t._reward_ref_refresh_steps
        if step_interval <= 0:
            return False
        if state.global_step < t._reward_ref_warmup_steps:
            return False
        if state.global_step == t._last_step_refreshed:
            return False
        return bool(state.global_step % step_interval == 0)

    def _should_refresh_epoch(self, state: Any) -> bool:
        t = self._trainer
        if t is None:
            return False
        if t._reward_ref_refresh_epochs <= 0:
            return False
        if t._last_epoch_refreshed is None:
            # First epoch end always refreshes if step-based refresh is off.
            return bool(state.epoch is not None)
        return bool(state.epoch is not None and state.epoch > t._last_epoch_refreshed)

    def _do_refresh(self, state: Any) -> None:
        t = self._trainer
        if t is None:
            return
        before_epoch = t._last_epoch_refreshed
        before_step = t._last_step_refreshed
        refreshed = t.refresh_reward_ref()
        if not refreshed:
            return
        if before_epoch is None:
            t._last_epoch_refreshed = state.epoch
        t._last_step_refreshed = state.global_step
        if IS_MAIN:
            print(
                f"RewardRefresher: snapshot updated at "
                f"epoch={state.epoch} step={state.global_step} "
                f"(prev epoch={before_epoch} prev step={before_step})."
            )

    def on_epoch_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        if self._should_refresh_epoch(state):
            self._do_refresh(state)

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        if self._should_refresh_step(state):
            self._do_refresh(state)


_rewards_refresher = RewardRefresher()


_model_device = next(model.parameters()).device
_dummy_ids = torch.tensor([tokenizer("Hello world")["input_ids"]], device=_model_device)
_dummy_mask = torch.ones_like(_dummy_ids)
_dummy_rewards = torch.zeros_like(_dummy_ids, dtype=TORCH_DTYPE)
with torch.no_grad():
    _out = model(
        input_ids=_dummy_ids,
        attention_mask=_dummy_mask,
        reward_values=_dummy_rewards,
    )
if IS_MAIN:
    print("Reward-values forward pass OK.")
    print(f"Logits shape: {_out.logits.shape}")
    # Reward-path probe: one Taichi delta step, so the reward kernel
    # compiles and runs even when the local model has no parallel
    # reward branch attached.
    _probe_state = torch.zeros(1, 4, 4, device=_model_device)
    _probe_out = delta_core_step(
        _probe_state,
        torch.randn(1, 4, device=_model_device),
        torch.randn(1, 4, device=_model_device),
        torch.full((1, 1), 0.5, device=_model_device),
        torch.full((1, 1), 0.5, device=_model_device),
        torch.rand(1, 4, device=_model_device),
        torch.rand(1, 4, device=_model_device),
    )
    print(f"Taichi delta probe : {tuple(_probe_out.shape)}")
    del _probe_state, _probe_out
del _dummy_ids, _dummy_mask, _dummy_rewards, _out, _model_device
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown.16]
# ## Training

# %% [code.17]
if _USE_SMOKE_CONFIG:
    # CPU smoke tests: plain AdamW (bitsandbytes is a CUDA dependency).
    _training_optim = "adamw_torch"
else:
    import bitsandbytes

    _ = bitsandbytes.__version__

    _training_optim = "paged_adamw_8bit"


def _stage_training_args(stage_name: str | None = None) -> TrainingArguments:
    # Reused for each curriculum stage; stage_name only affects run_name for TB.
    run_name = f"curriculum-{stage_name}" if stage_name else None
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_TRAIN_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        push_to_hub=PUSH_TO_HUB and IS_MAIN,
        hub_model_id=HUB_MODEL_ID or None,
        hub_token=HF_TOKEN,
        bf16=(TORCH_DTYPE == torch.bfloat16),
        fp16=(TORCH_DTYPE == torch.float16),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=_training_optim,
        report_to="none" if (_USE_SMOKE_CONFIG or not IS_MAIN) else ["tensorboard"],
        run_name=run_name,
        seed=42,
        data_seed=42,
        # PEFT + GatedRewardNet can leave sub-graphs unused on some steps under DDP.
        ddp_find_unused_parameters=True,
        dataloader_drop_last=True,
    )


if _USE_SMOKE_CONFIG:
    training_args = _stage_training_args(None)
    trainer = CPTSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        reward_every_n_steps=REWARD_EVERY_N_STEPS,
        reward_ref_refresh_epochs=REWARD_REFRESH_EVERY_N_EPOCHS,
        reward_ref_refresh_steps=REWARD_REFRESH_EVERY_N_STEPS,
        reward_ref_warmup_steps=REWARD_REFRESH_WARMUP_STEPS,
    )
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
elif USE_CURRICULUM:
    # Curriculum: iterate stages, building+tokenizing one stage at a time,
    # training, then dropping its snapshot caches before the next stage.
    _last_checkpoint: str | None = RESUME_FROM_CHECKPOINT
    trainer = None  # type: ignore[assignment]
    for _stage_idx, _stage_name in enumerate(_active_keys):  # type: ignore[name-defined]
        _stage_datasets = CURRICULUM_STAGES[_stage_name]
        if IS_MAIN:
            print(f"\n{'='*60}\n[stage {_stage_idx+1}/{len(_active_keys)}] {_stage_name}: {len(_stage_datasets)} datasets\n{'='*60}")
            for _n in _stage_datasets:
                print(f"  - {_n}")
        # Build and tokenize only this stage's datasets; previous stage's caches
        # have already been wiped via _drop_stage_cache.
        _stage_ds = peek_streaming_dataset(_stage_datasets)
        _stage_cols = _cols_to_remove_for(_stage_ds)
        _stage_ds = _stage_ds.map(tokenize_fn, batched=False, remove_columns=_stage_cols)
        _stage_ds = _stage_ds.filter(_keep_tokenized)
        if IS_MAIN:
            print(f"[stage {_stage_name}] tokenized, total stages remaining disk freed after prior stage")
        training_args = _stage_training_args(_stage_name)
        trainer = CPTSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=_stage_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            reward_every_n_steps=REWARD_EVERY_N_STEPS,
            reward_ref_refresh_epochs=REWARD_REFRESH_EVERY_N_EPOCHS,
            reward_ref_refresh_steps=REWARD_REFRESH_EVERY_N_STEPS,
            reward_ref_warmup_steps=REWARD_REFRESH_WARMUP_STEPS,
        )
        trainer.train(resume_from_checkpoint=_last_checkpoint)
        # Drop this stage's heavy caches before next stage to keep peak disk low.
        # The model weights stay in memory; only the dataset snapshots are removed.
        _drop_stage_cache(_stage_datasets)
        # Resolve latest checkpoint for next stage resume (if any).
        _latest = find_latest_checkpoint(OUTPUT_DIR)
        _last_checkpoint = str(_latest) if _latest is not None else None
        del _stage_ds, _stage_cols
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ACCEL_STATE.wait_for_everyone()
        if IS_MAIN:
            print(f"[stage {_stage_name}] complete; next checkpoint: {_last_checkpoint}")
else:
    # Legacy single-pass over all 17 datasets (capped-full or non-curriculum override).
    training_args = _stage_training_args(None)
    trainer = CPTSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        reward_every_n_steps=REWARD_EVERY_N_STEPS,
        reward_ref_refresh_epochs=REWARD_REFRESH_EVERY_N_EPOCHS,
        reward_ref_refresh_steps=REWARD_REFRESH_EVERY_N_STEPS,
        reward_ref_warmup_steps=REWARD_REFRESH_WARMUP_STEPS,
    )
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

ACCEL_STATE.wait_for_everyone()

# %% [markdown.18]
# ## Checkpoint Export & Save
#
# The trainer has already saved PEFT adapter checkpoints to ``OUTPUT_DIR``
# every ``save_steps``. This cell optionally merges the LoRA weights into
# the base model and saves the full model so it can be loaded for inference
# without PEFT.

# %% [code.19]
if IS_MAIN:
    print(f"\nCheckpoints saved to: {OUTPUT_DIR}")

    _FINAL_DIR = os.path.join(OUTPUT_DIR, "merged-final")
    _PEFT_DIR = os.path.join(OUTPUT_DIR, "peft-final")
    _merged_for_hub = None
    if MERGE_LORA_AFTER_TRAINING:
        print(f"Merging LoRA and saving full model to: {_FINAL_DIR}")
        merged = model.merge_and_unload()  # pyrefly: ignore[not-callable]
        merged.save_pretrained(_FINAL_DIR)  # pyrefly: ignore[not-callable]
        tokenizer.save_pretrained(_FINAL_DIR)
        print(
            f"Full model saved to {_FINAL_DIR} — all LoRA + directly trained weights fused, no loss"
        )
        _merged_for_hub = merged
    else:
        print(
            "MERGE_LORA_AFTER_TRAINING=False; saving adapter and also full fused model for verification"
        )
        model.save_pretrained(_PEFT_DIR)
        tokenizer.save_pretrained(_PEFT_DIR)
        print(f"Adapter saved to {_PEFT_DIR}")
        try:
            merged_local = model.merge_and_unload()  # pyrefly: ignore[not-callable]
            _merged_local_dir = os.path.join(OUTPUT_DIR, "merged-final-local")
            merged_local.save_pretrained(_merged_local_dir)  # pyrefly: ignore[not-callable]
            tokenizer.save_pretrained(_merged_local_dir)
            print(
                f"Full fused model also saved to {_merged_local_dir} for no-loss verification"
            )
            _merged_for_hub = merged_local
        except (OSError, RuntimeError) as exc:
            print(f"Warning: could not save merged full model locally: {exc}")
            _merged_for_hub = None

    if PUSH_TO_HUB:
        print(f"Pushing to Hub: {HUB_MODEL_ID}")
        # Push the fused full model when available, otherwise the PEFT adapter
        target_model = _merged_for_hub if _merged_for_hub is not None else model
        target_model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)  # pyrefly: ignore[not-callable]
        tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
        print("Push complete.")

ACCEL_STATE.wait_for_everyone()

# %% [markdown.20]
# ## Dataset Summary

# %% [code.21]
REPORT = [
    (
        "DylanRiden/smb-worldmodel-data",
        "~118k .npz frames (repo download; approximate)",
        "Downloaded repo, extracted smb_frames.zip, serialized 8-button action vectors to text.",
    ),
    (
        "Kalso42/WorldModelForMaze",
        "code repo + maze .txt files (approximate)",
        "Cloned repo via snapshot_download, read maze text files under data/**/*.txt as text.",
    ),
    (
        "ultrastar111/sokoban_...",
        "train split, viewer-enabled (approximate)",
        "Flattened messages JSON string to chat text; included task/seed/env_id metadata.",
    ),
    (
        "thuml/bytesized32-world-model-cot",
        "~301k train / ~2.9k test (approximate)",
        "Combined prompt list + reward_model JSON + extra_info JSON into text.",
    ),
    (
        "PatronusAI/world_model_corpus",
        "~239k train, config=train (approximate)",
        "Flattened messages chat list (tool-use traces with system prompts).",
    ),
    (
        "schema-harness/arc-agi-3-schema-traces",
        "~50 rows, test split (approximate)",
        "Serialized ARC benchmark metadata: task ID, status, win_levels, level0–level9 scores.",
    ),
    (
        "laion/strategic_game_chess",
        "train split (approximate)",
        "Converted SAN Moves list + Termination + Result into PGN-like text.",
    ),
    (
        "ryanmarten/OpenThoughts-1k-sample",
        "~1k samples (approximate)",
        "Flattened system + conversations chat into text.",
    ),
    (
        "Decix/ReBel-ALFWorld-SFT-Trajectories",
        "~426 rows (approximate)",
        "Parsed steps JSON string, serialized observation/action transcript.",
    ),
    (
        "greghavens/kimi-k3-coding-and-debugging-traces",
        "~3.9k rows (approximate)",
        "Flattened messages chat list with reasoning_content.",
    ),
    (
        "cot-leaderboard/cot-eval-traces-2.0",
        "~3.7M rows, test split, subsetted (approximate)",
        "Combined passage + question + options + answer + reasoning_trace into text.",
    ),
    (
        "Lichess/standard-chess-games",
        "~7.1B rows, subsetted (approximate)",
        "Converted movetext PGN + metadata (Event, White, Black, ECO, Opening) to text.",
    ),
    (
        "lockon/ToolACE",
        "train split (approximate)",
        "Flattened system + conversations chat into text.",
    ),
    (
        "faunix/Qwen3.8-27B-Distillation-40K",
        "~40k rows (approximate)",
        "Flattened messages chat list (distilled reasoning traces) + domain/category metadata.",
    ),
    (
        "Glint-Research/Fable-5-traces",
        "pi_agent config, train split (approximate)",
        "Flattened messages + prompt + trace into text.",
    ),
    (
        "Salesforce/wikitext",
        "wikitext-103-raw-v1: ~36k train, subsetted (approximate)",
        "Used single text column; empty rows replaced with [EMPTY_WIKITEXT_ROW].",
    ),
    (
        "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation",
        "~22.8M rows total, subsetted (approximate)",
        "Parsed messages_json string, flattened chat into text + task_type/source/domain metadata.",
    ),
]

if IS_MAIN:
    print("\n" + "=" * 60)
    print("DATASET REPORT — CPT Mixer")
    print("=" * 60)

    for name, size, usage in REPORT:
        print(f"\nDataset : {name}")
        print(f"  Size  : {size}")
        print(f"  Usage : {usage}")

    print("\n" + "=" * 60)
    print(f"Total datasets in mixer: {len(REPORT)}")
    print("=" * 60)
