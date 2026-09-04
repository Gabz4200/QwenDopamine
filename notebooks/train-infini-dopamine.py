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

# %% [code.1]
# %pip install --quiet \
#   "unsloth>=2024.7" \
#   "transformers>=5.0.0" \
#   "torch>=2.9.0" \
#   "trl>=0.12.0" \
#   "peft>=0.7.0" \
#   "datasets>=2.20.0" \
#   "accelerate>=0.34.0" \
#   "bitsandbytes>=0.43.0" \
#   "sentencepiece" \
#   "tokenizers" \
#   "einops" \
#   "tensorboard" \
#   "git+https://github.com/Gabz4200/QwenDopamine.git" \
#   "Pillow" \
#   "numpy" \
#   "tqdm"


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
from datasets import IterableDataset, interleave_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from qwendopamine.integrations.huggingface import HFIntegration
from qwendopamine.models.infinidopamine import (
    InfiniDopamineConfig,
    InfiniDopamineForConditionalGeneration,
    InfiniDopamineTextConfig,
    InfiniDopamineVisionConfig,
)

HFIntegration.register_infinidopamine_hf()

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

print("InfiniDopamine registered.")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device         : {torch.cuda.get_device_name(i)}")

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
]

DATASET_TEXT_COLUMN: str = "text"
MAX_SEQ_LENGTH: int = 1024

TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LOAD_IN_4BIT: bool = False

USE_LORA: bool = True
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
USE_RSLORA: bool = True
LORA_TARGET_MODULES = [
    "lm_head",
    "embed_tokens",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "in_proj_w",
    "in_proj_gate",
    "output_proj",
    "merger.linear_fc1",
    "merger.linear_fc2",
    "lm_head.dense",
    "lm_head.decoder",
    "delta_layer.q_proj",
    "delta_layer.memory_core.k_proj",
    "delta_layer.memory_core.v_proj",
    "delta_layer.memory_core.w_proj",
    "delta_layer.memory_core.e_proj",
    "advantage_gate.advantage_proj",
    "baseline_tracker.alpha_proj",
    "reward_gate_proj",
    "reward_branch_norm",
    "reward_branch.output_proj",
    "reward_branch.delta_layer.q_proj",
    "reward_branch.delta_layer.memory_core.k_proj",
    "reward_branch.delta_layer.memory_core.v_proj",
    "reward_branch.delta_layer.memory_core.w_proj",
    "reward_branch.delta_layer.memory_core.e_proj",
    "reward_branch.delta_layer.baseline_tracker.alpha_proj",
    "reward_branch.delta_layer.advantage_gate.advantage_proj",
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

PER_DEVICE_TRAIN_BATCH_SIZE: int = 1
GRADIENT_ACCUMULATION_STEPS: int = 16
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 0.01
LR_SCHEDULER_TYPE: str = "cosine"
WARMUP_STEPS: int = 100
NUM_TRAIN_EPOCHS: int = 1
LOGGING_STEPS: int = 10
SAVE_STEPS: int = 500
SAVE_TOTAL_LIMIT: int = 2

OUTPUT_DIR: str = f"./infini-dopamine-cpt-{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d-%H%M%S')}"
RESUME_FROM_CHECKPOINT: str | None = None
HUB_MODEL_ID: str = os.environ.get("HUB_MODEL_ID", "")
PUSH_TO_HUB: bool = bool(HUB_MODEL_ID)
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
MERGE_LORA_AFTER_TRAINING: bool = True

SMB_CACHE_DIR: str = "./smb-cache"
MAZE_CACHE_DIR: str = "./maze-cache"

MAX_ROWS_PER_DATASET: int | None = None
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
def _as_obj(d: dict) -> Any:
    class _Cfg:
        pass

    c = _Cfg()
    for k, v in d.items():
        setattr(c, k, v)
    return c


def build_text_config_from_qwen(qwen_cfg: Any) -> InfiniDopamineTextConfig:
    src = getattr(qwen_cfg, "text_config", qwen_cfg)
    if isinstance(src, dict):
        src = _as_obj(src)

    cfg_kwargs = dict(  # noqa: C408  # noqa: C408
        vocab_size=getattr(src, "vocab_size", 248320),
        hidden_size=getattr(src, "hidden_size", 1024),
        intermediate_size=getattr(src, "intermediate_size", 3584),
        num_hidden_layers=getattr(src, "num_hidden_layers", 24),
        num_key_value_heads=getattr(src, "num_key_value_heads", 2),
        max_position_embeddings=getattr(src, "max_position_embeddings", 262144),
        rms_norm_eps=getattr(src, "rms_norm_eps", 1e-6),
        rope_theta=getattr(src, "rope_theta", 10_000_000.0),
        attention_bias=getattr(src, "attention_bias", False),
        hidden_act=getattr(src, "hidden_act", "silu"),
        sliding_window=getattr(src, "sliding_window", None),
        attn_output_gate=getattr(src, "attn_output_gate", True),
        full_attention_interval=getattr(src, "full_attention_interval", 4),
        linear_conv_kernel_dim=getattr(src, "linear_conv_kernel_dim", 4),
        linear_key_head_dim=getattr(src, "linear_key_head_dim", 128),
        linear_num_key_heads=getattr(src, "linear_num_key_heads", 16),
        linear_num_value_heads=getattr(src, "linear_num_value_heads", 16),
        linear_value_head_dim=getattr(src, "linear_value_head_dim", 128),
        mlp_only_layers=getattr(src, "mlp_only_layers", []),
        mamba_ssm_dtype=getattr(src, "mamba_ssm_dtype", "float32"),
        # Parallel GatedRewardNet branch opt-in.
        use_parallel_reward=USE_PARALLEL_REWARD,
        parallel_reward_layers=PARALLEL_REWARD_LAYERS,
        reward_gate_init_bias=REWARD_GATE_INIT_BIAS,
        reward_memory_rank=REWARD_MEMORY_RANK,
        parallel_reward_gate_loss_weight=PARALLEL_REWARD_GATE_LOSS_WEIGHT,
    )
    layer_types = getattr(src, "layer_types", None)
    if layer_types is not None:
        cfg_kwargs["layer_types"] = layer_types

    rope_parameters = getattr(src, "rope_parameters", None)
    if rope_parameters is not None:
        cfg_kwargs["rope_parameters"] = rope_parameters

    return InfiniDopamineTextConfig(**cfg_kwargs)


def build_vision_config_from_qwen(qwen_cfg: Any) -> InfiniDopamineVisionConfig:
    src = getattr(qwen_cfg, "vision_config", qwen_cfg)
    if isinstance(src, dict):
        src = _as_obj(src)

    cfg_kwargs = dict(  # noqa: C408  # noqa: C408
        hidden_size=getattr(src, "hidden_size", 1280),
        in_channels=getattr(src, "in_channels", 3),
        intermediate_size=getattr(src, "intermediate_size", 3072),
        num_heads=getattr(src, "num_heads", 12),
        num_position_embeddings=getattr(src, "num_position_embeddings", 2304),
        out_hidden_size=getattr(src, "out_hidden_size", 1024),
        patch_size=getattr(src, "patch_size", 16),
        spatial_merge_size=getattr(src, "spatial_merge_size", 2),
        temporal_patch_size=getattr(src, "temporal_patch_size", 2),
        hidden_act=getattr(src, "hidden_act", "gelu_pytorch_tanh"),
        initializer_range=getattr(src, "initializer_range", 0.02),
    )
    return InfiniDopamineVisionConfig(**cfg_kwargs)


tokenizer: Any = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

try:
    processor: Any = AutoProcessor.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=True
    )
except (OSError, ValueError) as e:
    print(f"[WARN] AutoProcessor fallback to AutoTokenizer: {e}")
    processor = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

qwen_cfg = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

text_cfg = build_text_config_from_qwen(qwen_cfg)
vision_cfg = build_vision_config_from_qwen(qwen_cfg)
infini_cfg = InfiniDopamineConfig(
    text_config=text_cfg,
    vision_config=vision_cfg,
    image_token_id=getattr(qwen_cfg, "image_token_id", 248056),
    video_token_id=getattr(qwen_cfg, "video_token_id", 248057),
    vision_start_token_id=getattr(qwen_cfg, "vision_start_token_id", 248053),
    vision_end_token_id=getattr(qwen_cfg, "vision_end_token_id", 248054),
)

model = InfiniDopamineForConditionalGeneration(infini_cfg)

print("Loading base model weights...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
except (OSError, ValueError) as e:
    print(f"[WARN] AutoModelForCausalLM fallback to AutoModelForVision2Seq: {e}")
    try:
        from transformers import (
            AutoModelForVision2Seq,  # pyrefly: ignore[missing-module-attribute]
        )
    except ImportError as ie:
        raise ImportError(
            "Neither AutoModelForCausalLM nor AutoModelForVision2Seq is available"
        ) from ie

    base_model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

missing, unexpected = model.load_qwen35_weights(base_model, strict=False)

del base_model
gc.collect()
torch.cuda.empty_cache()

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
lora_cfg = None
if USE_LORA:
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        use_rslora=USE_RSLORA,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()


# Ensure ALL weights are trainable, including newly initialized ones
# that are not present in the Qwen3.5 checkpoint.
def ensure_all_trainable(model: Any, missing_keys: list[str]) -> None:
    """Set requires_grad=True for all newly initialized parameters.

    InfiniDopamine contains GDN-2 / reward-net modules that are not
    present in Qwen3.5. Their weights are initialized from scratch
    and must be trained. LoRA only covers nn.Linear modules, so we
    explicitly unfreeze any missing non-linear parameters too.
    """
    missing_set = set(missing_keys)
    unfrozen = 0
    for name, param in model.named_parameters():
        if name in missing_set and not param.requires_grad:
            param.requires_grad = True
            unfrozen += 1
    print(f"Unfrozen {unfrozen} newly initialized parameters for training.")


# Unfreeze newly initialized weights after LoRA is applied.
ensure_all_trainable(model, missing)

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
def _flatten_messages(messages: Any) -> str:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
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
        try:
            steps = json.loads(steps_raw)
        except json.JSONDecodeError as e:
            print(f"[WARN] Dataset formatting error: {e}")
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
        try:
            raw = json.loads(raw_value)
        except json.JSONDecodeError as e:
            print(f"[WARN] Dataset formatting error: {e}")
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
def load_smb_dataset() -> IterableDataset:
    import zipfile

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    repo_id = "DylanRiden/smb-worldmodel-data"
    cache_dir = Path(SMB_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename="smb_frames.zip",
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    except HfHubHTTPError as e:
        raise RuntimeError(f"Failed to download SMB dataset: {e}") from e

    extract_dir = cache_dir / "smb_frames"
    if not extract_dir.exists():
        print(f"Extracting {zip_path} -> {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    npz_files = sorted(extract_dir.rglob("*.npz"))
    print(f"Found {len(npz_files)} SMB .npz files")

    def _gen() -> Iterator[dict]:
        for npz_path in npz_files:
            try:
                data = np.load(npz_path)
                action = data["action"]
                if action.ndim == 0:
                    action = np.array([0.0] * 8)
                buttons = ["Up", "Down", "Left", "Right", "A", "B", "Start", "Select"]
                action_str = ", ".join(
                    f"{b}={float(v):.1f}" for b, v in zip(buttons, action.flatten()[:8])
                )
                yield {"text": f"SMB Frame Action: [{action_str}]"}
            except (OSError, KeyError, ValueError) as e:
                print(f"[WARN] Dataset formatting error: {e}")
                continue

    return IterableDataset.from_generator(_gen, gen_kwargs={})


def load_maze_dataset() -> IterableDataset:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    cache_dir = Path(MAZE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        maze_dir = snapshot_download(
            repo_id="Kalso42/WorldModelForMaze",
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    except HfHubHTTPError as e:
        raise RuntimeError(f"Failed to download maze dataset: {e}") from e
    maze_path = Path(maze_dir)
    txt_files = sorted(maze_path.glob("data/**/*.txt"))
    if not txt_files:
        txt_files = sorted(maze_path.rglob("*.txt"))
    print(f"Found {len(txt_files)} maze .txt files")

    def _gen() -> Iterator[dict]:
        for tf in txt_files:
            try:
                text = tf.read_text(encoding="utf-8", errors="replace")
                text = text.strip()
                if text:
                    yield {"text": text}
            except (OSError, UnicodeDecodeError) as e:
                print(f"[WARN] Dataset formatting error: {e}")
                continue

    return IterableDataset.from_generator(_gen, gen_kwargs={})


DATASET_CONFIG_MAP = {
    "PatronusAI/world_model_corpus": ("train", "train"),
    "Glint-Research/Fable-5-traces": ("pi_agent", "train"),
    "Salesforce/wikitext": ("wikitext-103-raw-v1", "train"),
    "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation": ("sft_balanced", "train"),
}

DATASET_SUBSET_MAP = {
    "Lichess/standard-chess-games": LICHESS_MAX_ROWS,
    "cot-leaderboard/cot-eval-traces-2.0": COT_EVAL_MAX_ROWS,
    "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation": R0B0TLAB_MAX_ROWS,
    "Salesforce/wikitext": WIKITEXT_MAX_ROWS,
}


def _apply_subset(ds: Any, dataset_name: str) -> IterableDataset:
    max_rows = DATASET_SUBSET_MAP.get(dataset_name)
    if max_rows is None:
        result: IterableDataset = ds
        return result

    def _gen() -> Iterator[dict]:
        for count, ex in enumerate(ds):
            yield ex
            if count >= max_rows:
                break

    return IterableDataset.from_generator(_gen, gen_kwargs={})


def build_streaming_dataset(
    dataset_names: list[str],
    seed: int = 42,
) -> IterableDataset:
    streams = []

    for name in dataset_names:
        if name == "DylanRiden/smb-worldmodel-data":
            streams.append(load_smb_dataset())
        elif name == "Kalso42/WorldModelForMaze":
            streams.append(load_maze_dataset())
        else:
            cfg, split = DATASET_CONFIG_MAP.get(name, ("default", "train"))
            ds = load_dataset(name, config=cfg, split=split, streaming=True)
            formatter = DATASET_FORMATTERS.get(name)
            if formatter is not None:
                ds = ds.map(
                    lambda ex, name=name: format_example(ex, name),  # pyrefly: ignore[implicit-any-lambda]
                    batched=False,
                )
            ds = _apply_subset(ds, name)
            streams.append(ds)

    if len(streams) == 1:
        return streams[0]

    result_ds: IterableDataset = interleave_datasets(
        streams,
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    return result_ds


def peek_streaming_dataset(dataset_names: list[str], seed: int = 42) -> IterableDataset:
    train_dataset = build_streaming_dataset(dataset_names, seed=seed)
    sample = next(iter(train_dataset))
    print(f"Sample keys  : {list(sample.keys())}")
    print(f"Sample text  : {str(sample.get('text', ''))[:240]}")
    print(f"Sample length: {len(str(sample.get('text', '')))}")
    del sample
    gc.collect()
    result: IterableDataset = train_dataset
    return result


train_dataset = peek_streaming_dataset(CPT_DATASETS)


# %% [code.13]
def tokenize_fn(example: dict) -> dict:
    tok = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "labels": tok["input_ids"],
    }


cols_to_remove: list[str] = []
if hasattr(train_dataset, "column_names") and train_dataset.column_names is not None:
    cols_to_remove = [
        c
        for c in train_dataset.column_names
        if c not in {"text", "input_ids", "attention_mask", "labels"}
    ]

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt",
)


train_dataset = train_dataset.map(
    tokenize_fn,
    batched=False,
    remove_columns=cols_to_remove,
)

print(f"Tokenized columns : {train_dataset.column_names}")
print(f"Max seq length    : {MAX_SEQ_LENGTH}")

# %% [markdown.14]
# ## Reward Conditioning, Parallel Branch & Custom Trainer
#
# Computes per-token pseudo-rewards via detached base-model loss and passes
# `reward_values` into `InfiniDopamine` during training.
#
# When `USE_PARALLEL_REWARD=True` (or `PARALLEL_REWARD_LAYERS` is non-empty)
# the trainer also exposes diagnostics for the parallel `GatedRewardNet`
# branch: gate mean/max, effective branch contribution, EMA value baseline,
# and fast-weight state norm. These are logged every
# `PARALLEL_REWARD_LOG_INTERVAL` steps so the dopamine branch stays
# observable in TensorBoard even before reward signals become meaningful.


# %% [code.15]
def build_reward_values(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token pseudo-rewards from a detached base-model pass.

    Alignment: when generating x(t+1) from input x(t), we consume
    the reward of token x(t). This is achieved by shifting rewards
    one position forward: reward_values[:, 1:] = rewards.
    """
    with torch.no_grad():
        base_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        shift_logits = base_outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(input_ids.size(0), -1)

        if REWARD_LOSS_TYPE in ("nll", "ce"):
            rewards = -token_loss * REWARD_SCALE
        else:
            rewards = (-token_loss).exp() * REWARD_SCALE

        reward_values = torch.zeros_like(input_ids, dtype=TORCH_DTYPE)
        reward_values[:, 1:] = rewards.to(TORCH_DTYPE)
        reward_values = reward_values * attention_mask
    return reward_values


class CPTSFTTrainer(SFTTrainer):
    def create_optimizer(self) -> Any:
        """Create optimizer with lower LR for embedding layers, per Unsloth CPT guidance."""
        if hasattr(self, "optimizer") and self.optimizer is not None:
            return self.optimizer

        embed_param_names = {"embed_tokens", "lm_head"}
        embed_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in embed_param_names):
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

        optimizer_cls = self.get_optimizer_cls()
        result: Any = optimizer_cls(param_groups, **self.optimizer_kwargs)
        return result

    def __init__(self, *args, reward_every_n_steps: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reward_every_n_steps = max(1, int(reward_every_n_steps))
        self._global_step = 0

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
            reward_values = build_reward_values(input_ids, attention_mask)
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


dummy_ids = torch.tensor([tokenizer("Hello world")["input_ids"]], device="cpu")
dummy_mask = torch.ones_like(dummy_ids)
dummy_rewards = torch.zeros_like(dummy_ids, dtype=TORCH_DTYPE)
with torch.no_grad():
    out = model(
        input_ids=dummy_ids,
        attention_mask=dummy_mask,
        reward_values=dummy_rewards,
    )
print("Reward-values forward pass OK.")
print(f"Logits shape: {out.logits.shape}")

# %% [markdown.16]
# ## Training

# %% [code.17]
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=HUB_MODEL_ID or None,
    hub_token=HF_TOKEN,
    bf16=(TORCH_DTYPE == torch.bfloat16),
    fp16=(TORCH_DTYPE == torch.float16),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    report_to=["tensorboard"],
    seed=42,
    data_seed=42,
)

trainer = CPTSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    reward_every_n_steps=REWARD_EVERY_N_STEPS,
)

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

# %% [markdown.18]
# ## Checkpoint Export & Dataset Summary

# %% [code.19]
print("\n" + "=" * 60)
print("DATASET REPORT — CPT Mixer")
print("=" * 60)

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

for name, size, usage in REPORT:
    print(f"\nDataset : {name}")
    print(f"  Size  : {size}")
    print(f"  Usage : {usage}")

print("\n" + "=" * 60)
print(f"Total datasets in mixer: {len(REPORT)}")
print("=" * 60)
