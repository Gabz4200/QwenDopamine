"""Kaggle 2xT4 GPU smoke-test + FineWeb-Edu Micro training comparison: GatedDeltaNet-2 (GDN-2) vs. GatedSurpriseNet (1.3B).

Architecture Comparison:
  - GDN-2: Multi-Head Pure Recurrent Decoder (24 layers, 16 heads, head_size=128, hidden_dim=2048)
    using decoupled channel-wise erase gate b_t in [0, 3.0] and write gate w_t in [0, 1.5].
  - GatedSurpriseNet: Multi-Head Pure Recurrent Decoder (24 layers, 16 heads, head_size=128, hidden_dim=2048)
    using Precision-Weighted Surprise Memory (pi_t in [0, 2.0], b_t in [0, 3.0], w_t in [0, 1.5]).

Part 1 — Architecture Inspection & Sanity Checks (DRY reused infrastructure)
  1. Inspect parameter specifications for both 1.3B models.
  2. Synthetic overfit sanity checks for GDN-2 and GatedSurpriseNet.
  3. Recurrence & chunkwise scan parity checks for both architectures.

Part 2 — FineWeb-Edu Micro LM Training & Comparative Analysis
  Train both models on bhavnicksm/fineweb-edu-micro (1M tokens) using:
    - 8-bit AdamW + gradient checkpointing + AMP + DDP + early stopping
    - Log train/val loss, perplexity, negative log-likelihood, and throughput (tok/s).

Plots Output:
  - `metrics_gdn2.png`: Standalone metrics for GDN-2
  - `metrics_gated_surprise_net.png`: Standalone metrics for GatedSurpriseNet
  - `metrics_comparison_gdn2_vs_surprise.png`: Side-by-side comparative curves

Designed for Kaggle with 2x T4 GPUs (16GB each). Uses ``torch.distributed`` DDP
when multiple GPUs are detected; falls back to single-GPU/CPU otherwise.

Run on Kaggle (2x T4 GPUs via DDP)::

    torchrun --nproc_per_node=2 notebooks/test_gated_surprise_net_vs_gdn2_gpu.py

or single GPU / CPU::

    python notebooks/test_gated_surprise_net_vs_gdn2_gpu.py
"""

from __future__ import annotations

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import fcntl
import math
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Always prioritize local repository src/ directory over site-packages
try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
except (NameError, TypeError):
    _REPO_ROOT = Path.cwd()
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib import ticker
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from transformers import AutoTokenizer

_LOCK_PATH = os.path.join(tempfile.gettempdir(), "qwendopamine_gpu_setup.lock")


def _ensure_dependencies() -> None:
    def _check() -> tuple[bool, bool]:
        need_mask = False
        try:
            import transformers.masking_utils

            need_mask = not hasattr(
                transformers.masking_utils, "create_recurrent_attention_mask"
            )
        except (ImportError, AttributeError):
            need_mask = True

        need_bnb = False
        if torch.cuda.is_available():
            try:
                import bitsandbytes  # noqa: F401

                need_bnb = False
            except ImportError:
                need_bnb = True

        return need_mask, need_bnb

    need_mask, need_bnb = _check()
    if not (need_mask or need_bnb):
        return

    with open(_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            need_mask, need_bnb = _check()
            if not (need_mask or need_bnb):
                return

            to_install: list[str] = []
            if need_mask:
                to_install.append("transformers>=4.49.0")
            if need_bnb:
                to_install.append("bitsandbytes>=0.41.0")

            if to_install:
                print(
                    "[setup] Installing/upgrading dependencies "
                    f"({', '.join(to_install)})..."
                )
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--upgrade-strategy",
                            "only-if-needed",
                        ]
                        + to_install,
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"[setup] Warning: dependency installation returned non-zero exit code ({e}). "
                        "Continuing with existing packages."
                    )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


_ensure_dependencies()

from qwendopamine.models.gated_surprise_net import (
    SurpriseMemory,
    l2_normalize_last,
)
from qwendopamine.models.gdn2.gdn2 import (
    torch_chunk_gdn2,
    torch_recurrent_gdn2,
)
from qwendopamine.models.surprise_gpt import (
    SurpriseGPT,
    SurpriseGPTConfig,
    compute_model_params,
)

GPT = SurpriseGPT
Config = SurpriseGPTConfig

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

has_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count() if has_cuda else 0

if has_cuda:
    torch.set_float32_matmul_precision("high")
    if WORLD_SIZE > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda", LOCAL_RANK)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    if WORLD_SIZE > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    device = torch.device("cpu")
    dtype = torch.float32

print(
    f"[env] rank={RANK}/{WORLD_SIZE}  local_rank={LOCAL_RANK}  "
    f"device={device}  dtype={dtype}"
)
if has_cuda:
    print(
        f"[env] GPU: {torch.cuda.get_device_name(LOCAL_RANK)} (total GPUs visible: {num_gpus})"
    )
    if num_gpus > 1 and WORLD_SIZE == 1:
        _script_name = (
            sys.argv[0]
            if (sys.argv and sys.argv[0])
            else "notebooks/test_gated_surprise_net_vs_gdn2_gpu.py"
        )
        print(
            f"[env] Note: {num_gpus} GPUs detected. To run DDP across both GPUs, launch with:\n"
            f"      torchrun --nproc_per_node={num_gpus} {_script_name}"
        )

IS_MAIN = RANK == 0


def inspect_architecture(mixer_type: str = "surprise") -> None:
    """Inspect and print model parameter breakdown for GDN-2 or GatedSurpriseNet."""
    name = "GDN-2" if mixer_type == "gdn2" else "GatedSurpriseNet"
    cfg = Config.from_name(
        "1B_mha",
        mixer_type=mixer_type,
        surprise_net_per_layer=1,
        train_chunk_size=128,
        use_short_conv=True,
        norm_eps=1e-6,
    )
    stats = compute_model_params(cfg)

    print("=" * 60)
    print(f"1.3B Scale Pure Recurrent Model Specification ({cfg.name} - {name})")
    print("=" * 60)
    print(f"  Total Parameters:        {stats['total']:,} (~{stats['total']/1e9:.2f}B)")
    print(f"  Layers (n_layer):        {cfg.n_layer}")
    print(f"  Hidden Dim (n_embd):     {cfg.n_embd}")
    print(f"  Recurrent Heads:         {cfg.n_head} (head_dim={cfg.head_size})")
    print(f"  SwiGLU Intermediate:     {cfg.intermediate_size} (~8/3 x hidden_dim)")
    print(f"  Vocabulary Size:         {cfg.vocab_size:,}")
    print(f"  Max Context Length:      {cfg.block_size:,}")
    print(f"  Recurrent Layers:        {stats['num_surprise_layers']}x {name} (All 24 layers)")
    print(f"  Self-Attention Layers:   {stats['num_standard_layers']} (Zero self-attention)")
    print(f"  Recurrent Block Params:  {stats['surprise_block']:,}")
    print(f"  Token Embeddings:        {stats['embed']:,}")
    print("=" * 60)


def run_synthetic_overfit(mixer_type: str = "surprise") -> tuple[float, float, bool]:
    """Verify learning on a small multi-head pure recurrent GPT model."""
    name = "GDN-2" if mixer_type == "gdn2" else "GatedSurpriseNet"
    torch.manual_seed(0)
    vocab_size = 64
    seq_len = 32
    batch_size = 4
    synth_steps = 200

    cfg = Config(
        name=f"synthetic_recurrent_{mixer_type}",
        block_size=seq_len,
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        head_size=32,
        n_query_groups=4,
        intermediate_size=344,
        norm_eps=1e-6,
        train_chunk_size=seq_len,
        surprise_net_per_layer=1,
        mixer_type=mixer_type,
    )
    model = GPT(cfg).to(device=device, dtype=dtype)

    if WORLD_SIZE > 1:
        ddp_kwargs = (
            {"device_ids": [LOCAL_RANK], "output_device": LOCAL_RANK}
            if has_cuda
            else {}
        )
        model = DDP(model, **ddp_kwargs)

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if IS_MAIN:
        print(f"[synth {name}] params: {total_params:,}  trainable: {trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    initial_loss = 0.0
    final_loss_val = 0.0
    for step in range(synth_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.item())
        final_loss_val = float(loss.item())
        if step % 50 == 0 and IS_MAIN:
            print(f"[synth {name}] step {step:03d}  loss={loss.item():.4f}")

    if IS_MAIN:
        print(f"[synth {name}] initial={initial_loss:.4f}  final={final_loss_val:.4f}")

    passed = final_loss_val < initial_loss * 0.5 and math.isfinite(final_loss_val)
    return initial_loss, final_loss_val, passed


@torch.no_grad()
def run_gdn2_parity() -> bool:
    """Verify validity and finiteness of multi-head GDN-2 recurrent scan."""
    bs, t, h, d_k, d_v = 2, 32, 4, 16, 16
    torch.manual_seed(42)
    q = torch.randn(bs, t, h, d_k, device=device, dtype=dtype)
    k = torch.randn(bs, t, h, d_k, device=device, dtype=dtype)
    v = torch.randn(bs, t, h, d_v, device=device, dtype=dtype)
    g = torch.randn(bs, t, h, d_k, device=device, dtype=dtype).abs_().mul_(-1)
    b = torch.rand(bs, t, h, d_k, device=device, dtype=dtype)
    w = torch.rand(bs, t, h, d_v, device=device, dtype=dtype)

    out_s, _ = torch_recurrent_gdn2(
        q, k, v, g, b, w, initial_state=None, output_final_state=True
    )
    # chunk_size=16 keeps cumulative log-decay to ~16 steps per chunk,
    # preventing gamma underflow (exp(-25) ≈ 1e-11) that makes kbar = k/gamma
    # numerically explosive. The stability bound is decay-magnitude-dependent,
    # not seq_len-dependent.
    out_c, _ = torch_chunk_gdn2(
        q, k, v, g, b, w, initial_state=None, output_final_state=True,
        chunk_size=16,
    )

    max_diff = (out_s.float() - out_c.float()).abs().max().item()
    passed = (
        torch.all(torch.isfinite(out_s)).item()
        and torch.all(torch.isfinite(out_c)).item()
        and torch.allclose(out_s, out_c, atol=1e-3)
    )
    if not passed and IS_MAIN:
        print(f"[parity GDN-2] serial vs chunk scan check: FAIL (max_diff={max_diff:.4e})")
    return bool(passed)


@torch.no_grad()
def run_surprise_parity() -> bool:
    """Verify parity between serial scan and chunk parallel scan for GatedSurpriseNet."""
    bs, t, h, d = 2, 32, 4, 16
    torch.manual_seed(42)
    q = l2_normalize_last(torch.randn(bs, t, h, d, device=device, dtype=dtype))
    k = l2_normalize_last(torch.randn(bs, t, h, d, device=device, dtype=dtype))
    v = torch.randn(bs, t, h, d, device=device, dtype=dtype)
    g = torch.randn(bs, t, h, d, device=device, dtype=dtype).abs_().mul_(-1)
    b = torch.rand(bs, t, h, d, device=device, dtype=dtype)
    w = torch.rand(bs, t, h, d, device=device, dtype=dtype)
    sigma_sq = F.softplus(torch.randn(bs, t, h, d, device=device, dtype=dtype)) + 1e-4

    memory = SurpriseMemory(num_heads=h, head_k_dim=d, head_v_dim=d).to(
        device=device, dtype=dtype
    )

    out_s, _, nll_s = memory.serial_scan(q, k, v, g, b, w, sigma_sq=sigma_sq)
    out_c, _, nll_c = memory.chunk_parallel_training_scan(
        q, k, v, g, b, w, sigma_sq=sigma_sq, chunk_size=16
    )

    passed = torch.allclose(out_s, out_c, atol=1e-3) and torch.allclose(
        nll_s, nll_c, atol=1e-3
    )
    if not passed and IS_MAIN:
        print("[parity SurpriseNet] serial vs chunk scan check: FAIL")
    return bool(passed)


@dataclass
class FineWebMicroConfig:
    dataset_name: str = "bhavnicksm/fineweb-edu-micro"
    max_seq_len: int = 512
    max_train_examples: int | None = None
    max_val_examples: int | None = None
    batch_size: int = 2
    val_split_ratio: float = 0.1
    seed: int = 42


def load_fineweb_micro_tokenized(
    cfg: FineWebMicroConfig, tokenizer: Any
) -> tuple[TensorDataset, TensorDataset]:
    """Load and tokenize passages from bhavnicksm/fineweb-edu-micro (shared DRY data loader)."""
    if IS_MAIN:
        print(f"[data] Loading dataset '{cfg.dataset_name}' ...")
    try:
        ds = load_dataset(cfg.dataset_name)
    except TypeError:
        ds = load_dataset(cfg.dataset_name, trust_remote_code=True)

    raw_train = ds["train"]
    split = raw_train.train_test_split(test_size=cfg.val_split_ratio, seed=cfg.seed)

    def encode_split(split_data: Any, max_examples: int | None) -> list[list[int]]:
        texts: list[str] = []
        for ex in split_data:
            txt = ex.get("text", "").strip()
            if txt:
                texts.append(txt)
            if max_examples is not None and len(texts) >= max_examples:
                break
        if IS_MAIN:
            print(f"[data] Loaded {len(texts)} educational passages")

        all_tokens: list[int] = []
        eos_id = tokenizer.eos_token_id or 50256
        for txt in texts:
            ids = tokenizer(txt, truncation=False, add_special_tokens=False)[
                "input_ids"
            ]
            all_tokens.extend(ids)
            all_tokens.append(eos_id)

        seqs: list[list[int]] = []
        for i in range(0, len(all_tokens) - cfg.max_seq_len, cfg.max_seq_len):
            seqs.append(all_tokens[i : i + cfg.max_seq_len + 1])
        return seqs

    train_seqs = encode_split(split["train"], cfg.max_train_examples)
    val_seqs = encode_split(split["test"], cfg.max_val_examples)

    if not train_seqs or not val_seqs:
        raise RuntimeError("Failed to generate tokenized sequences from dataset.")

    train_in = torch.tensor([s[:-1] for s in train_seqs], dtype=torch.long)
    train_tgt = torch.tensor([s[1:] for s in train_seqs], dtype=torch.long)
    val_in = torch.tensor([s[:-1] for s in val_seqs], dtype=torch.long)
    val_tgt = torch.tensor([s[1:] for s in val_seqs], dtype=torch.long)

    if IS_MAIN:
        print(
            f"[data] train sequences: {train_in.shape[0]} | "
            f"val sequences: {val_in.shape[0]} (seq_len={cfg.max_seq_len})"
        )
    return TensorDataset(train_in, train_tgt), TensorDataset(val_in, val_tgt)


@dataclass
class TrainConfig:
    num_steps: int = 1000
    log_interval: int = 1
    eval_interval: int = 50
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 50
    grad_clip: float = 1.0
    grad_accum_steps: int = 4
    use_amp: bool = True
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-3


def configure_optimizers(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    device_type: str,
) -> torch.optim.Optimizer:
    """Create optimizer applying decoupled weight decay exclusively to 2D weight matrices."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim >= 2
            and not getattr(param, "_no_weight_decay", False)
            and "norm" not in name.lower()
            and "bias" not in name.lower()
        ):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer_cls = torch.optim.AdamW
    if device_type == "cuda":
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.PagedAdamW8bit
            if IS_MAIN:
                print(
                    "[opt] Using bitsandbytes PagedAdamW8bit "
                    f"(decay={len(decay_params)} tensors, no_decay={len(no_decay_params)} tensors)"
                )
        except ImportError:
            if IS_MAIN:
                print("[opt] bitsandbytes not found, falling back to standard AdamW")

    return optimizer_cls(param_groups, lr=lr, betas=betas, eps=eps)


def evaluate(
    model: nn.Module,
    val_dl: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    eval_t0 = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(val_dl):
            if max_batches is not None and batch_idx >= max_batches:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda"),
            ):
                logits = model(xb)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            batch_tokens = yb.numel()
            total_loss += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens

    eval_time = time.perf_counter() - eval_t0

    if WORLD_SIZE > 1:
        stats = torch.tensor([total_loss, total_tokens], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_tokens = stats.tolist()

    avg_loss = total_loss / max(total_tokens, 1.0)
    perplexity = math.exp(min(avg_loss, 50))
    tps = total_tokens / max(eval_time, 1e-9)
    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
        "val_nll": avg_loss,
        "val_tokens": total_tokens,
        "val_tps": tps,
    }


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    model_name: str = "Model",
) -> dict[str, list[float]]:
    """Shared training runner for GDN-2 or GatedSurpriseNet."""
    model.train()

    optimizer = configure_optimizers(
        model=model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        device_type=device.type,
    )
    loss_fn = nn.CrossEntropyLoss()

    try:
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=(device.type == "cuda" and dtype == torch.float16 and cfg.use_amp),
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=(device.type == "cuda" and dtype == torch.float16 and cfg.use_amp)
        )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_steps": [],
        "val_loss": [],
        "val_perplexity": [],
        "val_nll": [],
        "val_tps": [],
        "val_steps": [],
        "tokens_seen": [],
        "lr": [],
        "step_time_s": [],
    }
    tokens_seen = 0
    t0 = time.perf_counter()
    step = 0
    epoch = 0
    accum_loss = 0.0
    steps_since_log = 0
    stop_training = False

    best_val_loss = float("inf")
    patience_counter = 0

    while step < cfg.num_steps and not stop_training:
        if isinstance(train_dl.sampler, DistributedSampler):
            train_dl.sampler.set_epoch(epoch)
        epoch += 1

        for xb, yb in train_dl:
            if step >= cfg.num_steps or stop_training:
                break

            xb = xb.to(device)
            yb = yb.to(device)

            if step < cfg.warmup_steps:
                lr = cfg.lr * (step + 1) / cfg.warmup_steps
            else:
                progress = (step - cfg.warmup_steps) / max(
                    cfg.num_steps - cfg.warmup_steps, 1
                )
                lr = cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda"),
            ):
                logits = model(xb)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
                loss = loss / cfg.grad_accum_steps

            is_accumulating = (
                (step + 1) % cfg.grad_accum_steps != 0
                and (step + 1) != cfg.num_steps
            )
            sync_ctx = (
                model.no_sync()
                if (WORLD_SIZE > 1 and is_accumulating and hasattr(model, "no_sync"))
                else nullcontext()
            )
            with sync_ctx:
                scaler.scale(loss).backward()

            accum_loss += float(loss.item()) * cfg.grad_accum_steps
            tokens_seen += yb.numel()
            steps_since_log += 1

            if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == cfg.num_steps:
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step_time = time.perf_counter() - t0

            if step % cfg.log_interval == 0 and IS_MAIN:
                avg_train_loss = accum_loss / max(1, steps_since_log)
                history["train_loss"].append(avg_train_loss)
                history["train_steps"].append(float(step))
                history["tokens_seen"].append(float(tokens_seen))
                history["lr"].append(lr)
                history["step_time_s"].append(step_time)
                print(
                    f"[{model_name} train] step {step:5d}  loss={avg_train_loss:.4f}  "
                    f"lr={lr:.2e}  tok/s={tokens_seen / max(step_time, 1e-9):.0f}"
                )
                accum_loss = 0.0
                steps_since_log = 0

            if step % cfg.eval_interval == 0 and step > 0:
                metrics = evaluate(
                    model, val_dl, loss_fn, device, dtype, max_batches=50
                )
                if IS_MAIN:
                    val_loss = metrics["val_loss"]
                    history["val_loss"].append(val_loss)
                    history["val_perplexity"].append(metrics["val_perplexity"])
                    history["val_nll"].append(metrics["val_nll"])
                    history["val_tps"].append(metrics["val_tps"])
                    history["val_steps"].append(float(step))
                    print(
                        f"[{model_name} eval]  step {step:5d}  loss={val_loss:.4f}  "
                        f"ppl={metrics['val_perplexity']:.2f}  "
                        f"nll={metrics['val_nll']:.2f}  "
                        f"tps={metrics['val_tps']:.0f}"
                    )

                    if val_loss < best_val_loss - cfg.early_stopping_min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= cfg.early_stopping_patience:
                            print(
                                f"[{model_name} early_stop] Validation loss plateaued for "
                                f"{patience_counter} evaluations (best={best_val_loss:.4f}). Stopping."
                            )
                            stop_training = True

                if WORLD_SIZE > 1:
                    stop_tensor = torch.tensor(
                        [1 if stop_training else 0], device=device
                    )
                    dist.broadcast(stop_tensor, src=0)
                    stop_training = bool(stop_tensor.item())

                if stop_training:
                    break
                model.train()

            step += 1

        if stop_training:
            break

    metrics = evaluate(model, val_dl, loss_fn, device, dtype)
    if IS_MAIN:
        history["val_loss"].append(metrics["val_loss"])
        history["val_perplexity"].append(metrics["val_perplexity"])
        history["val_nll"].append(metrics["val_nll"])
        history["val_tps"].append(metrics["val_tps"])
        history["val_steps"].append(float(step))
        print(
            f"[{model_name} final] loss={metrics['val_loss']:.4f}  "
            f"ppl={metrics['val_perplexity']:.2f}  "
            f"nll={metrics['val_nll']:.2f}  "
            f"tps={metrics['val_tps']:.0f}"
        )
    return history


def plot_single_model_metrics(
    history: dict[str, list[float]], title: str, save_path: str
) -> None:
    """Plot metric curves independently for a single model."""
    if not IS_MAIN:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=13)

    plots = [
        (axes[0, 0], "train_steps", "train_loss", "Train Loss", "Loss", "tab:blue"),
        (axes[0, 1], "val_steps", "val_loss", "Val Loss", "Loss", "tab:orange"),
        (
            axes[0, 2],
            "val_steps",
            "val_perplexity",
            "Val Perplexity",
            "Perplexity",
            "tab:green",
        ),
        (axes[1, 0], "val_steps", "val_nll", "Val NLL", "NLL", "tab:red"),
        (
            axes[1, 1],
            "val_steps",
            "val_tps",
            "Val Throughput",
            "Tokens / sec",
            "tab:purple",
        ),
        (axes[1, 2], "train_steps", "lr", "Learning Rate", "LR", "tab:brown"),
    ]

    for ax, x_key, y_key, subt, ylabel, color in plots:
        x_data = history.get(x_key, [])
        y_data = history.get(y_key, [])
        if x_data and y_data and len(x_data) == len(y_data):
            ax.plot(x_data, y_data, marker="o", color=color)
            ax.set_title(subt)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.grid(True)

    axes[1, 2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[plot] Saved single model metrics plot to {save_path}")
    plt.close(fig)


def plot_comparison_metrics(
    history_gdn2: dict[str, list[float]],
    history_surprise: dict[str, list[float]],
    save_path: str = "metrics_comparison_gdn2_vs_surprise.png",
) -> None:
    """Plot comparative metric curves for both GDN-2 and GatedSurpriseNet on the same axes."""
    if not IS_MAIN:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Comparative Training Metrics: GDN-2 vs. Precision-Weighted GatedSurpriseNet (1.3B)",
        fontsize=14,
        fontweight="bold",
    )

    plots = [
        (axes[0, 0], "train_steps", "train_loss", "Train Loss", "Loss"),
        (axes[0, 1], "val_steps", "val_loss", "Val Loss", "Loss"),
        (axes[0, 2], "val_steps", "val_perplexity", "Val Perplexity", "Perplexity"),
        (axes[1, 0], "val_steps", "val_nll", "Val NLL", "NLL"),
        (axes[1, 1], "val_steps", "val_tps", "Val Throughput", "Tokens / sec"),
        (axes[1, 2], "train_steps", "lr", "Learning Rate", "LR"),
    ]

    for ax, x_key, y_key, subt, ylabel in plots:
        x_g = history_gdn2.get(x_key, [])
        y_g = history_gdn2.get(y_key, [])
        x_s = history_surprise.get(x_key, [])
        y_s = history_surprise.get(y_key, [])

        if x_g and y_g and len(x_g) == len(y_g):
            ax.plot(
                x_g,
                y_g,
                marker="o",
                linestyle="-",
                color="tab:blue",
                label="GDN-2",
            )
        if x_s and y_s and len(x_s) == len(y_s):
            ax.plot(
                x_s,
                y_s,
                marker="s",
                linestyle="--",
                color="tab:red",
                label="GatedSurpriseNet",
            )

        ax.set_title(subt)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(loc="best")

    axes[1, 2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[plot] Saved comparative metrics plot to {save_path}")
    plt.close(fig)


def main() -> None:
    if IS_MAIN:
        print("\n" + "=" * 70)
        print("PART 1: ARCHITECTURE INSPECTION & SANITY CHECKS")
        print("=" * 70)
        inspect_architecture("gdn2")
        inspect_architecture("surprise")

    sanity_ok = True

    if IS_MAIN:
        print("\n--- Sanity Check 1: GDN-2 Synthetic Overfit ---")
    _, _, synth_gdn2_ok = run_synthetic_overfit("gdn2")
    if not synth_gdn2_ok:
        sanity_ok = False
        if IS_MAIN:
            print("[check] GDN-2 Synthetic overfit: FAIL")
    elif IS_MAIN:
        print("[check] GDN-2 Synthetic overfit: PASS")

    if IS_MAIN:
        print("\n--- Sanity Check 2: GatedSurpriseNet Synthetic Overfit ---")
    _, _, synth_surp_ok = run_synthetic_overfit("surprise")
    if not synth_surp_ok:
        sanity_ok = False
        if IS_MAIN:
            print("[check] GatedSurpriseNet Synthetic overfit: FAIL")
    elif IS_MAIN:
        print("[check] GatedSurpriseNet Synthetic overfit: PASS")

    if IS_MAIN:
        print("\n--- Sanity Check 3: GDN-2 Scan Parity ---")
    parity_gdn2_ok = run_gdn2_parity()
    if not parity_gdn2_ok:
        sanity_ok = False
        if IS_MAIN:
            print("[check] GDN-2 Parity: FAIL")
    elif IS_MAIN:
        print("[check] GDN-2 Parity: PASS")

    if IS_MAIN:
        print("\n--- Sanity Check 4: GatedSurpriseNet Scan Parity ---")
    parity_surp_ok = run_surprise_parity()
    if not parity_surp_ok:
        sanity_ok = False
        if IS_MAIN:
            print("[check] GatedSurpriseNet Parity: FAIL")
    elif IS_MAIN:
        print("[check] GatedSurpriseNet Parity: PASS")

    if WORLD_SIZE > 1:
        sanity_tensor = torch.tensor([1 if sanity_ok else 0], device=device)
        dist.broadcast(sanity_tensor, src=0)
        sanity_ok = bool(sanity_tensor.item())

    if IS_MAIN:
        if sanity_ok:
            print("\n[check] All 4 sanity checks passed! Proceeding to LM Training.")
        else:
            print("\n[check] One or more sanity checks failed. Skipping LM Training.")

    if not sanity_ok:
        if WORLD_SIZE > 1:
            dist.destroy_process_group()
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if IS_MAIN:
        print("\n" + "=" * 70)
        print("PART 2: FineWeb-Edu Micro 1.3B LM TRAINING & COMPARISON")
        print("=" * 70)

    data_cfg = FineWebMicroConfig(
        max_seq_len=512,
        batch_size=2,
        val_split_ratio=0.1,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 1_000_000
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_fineweb_micro_tokenized(data_cfg, tokenizer)

    train_sampler_gdn2 = DistributedSampler(train_ds, shuffle=True) if WORLD_SIZE > 1 else None
    val_sampler_gdn2 = DistributedSampler(val_ds, shuffle=False) if WORLD_SIZE > 1 else None

    pin_memory = device.type == "cuda"
    train_dl_gdn2 = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler_gdn2,
        shuffle=(train_sampler_gdn2 is None),
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    val_dl_gdn2 = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        sampler=val_sampler_gdn2,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    train_cfg = TrainConfig(
        num_steps=1000,
        log_interval=1,
        eval_interval=50,
        lr=3e-4,
        min_lr=3e-5,
        weight_decay=0.1,
        warmup_steps=50,
        grad_accum_steps=4,
        early_stopping_patience=4,
        early_stopping_min_delta=1e-3,
    )

    # 1. Train GDN-2 Model
    if IS_MAIN:
        print("\n--- Training Model 1/2: GDN-2 (1.3B Scale) ---")

    cfg_gdn2 = Config.from_name(
        "1B_mha",
        block_size=max(data_cfg.max_seq_len, 2048),
        vocab_size=tokenizer.vocab_size,
        padded_vocab_size=tokenizer.vocab_size,
        mixer_type="gdn2",
        surprise_net_per_layer=1,
        train_chunk_size=128,
        use_short_conv=True,
        norm_eps=1e-6,
        gradient_checkpointing=True,
    )
    model_gdn2 = GPT(cfg_gdn2).to(device=device, dtype=dtype)
    model_gdn2.gradient_checkpointing_enable()

    if WORLD_SIZE > 1:
        ddp_kwargs = (
            {"device_ids": [LOCAL_RANK], "output_device": LOCAL_RANK}
            if has_cuda
            else {}
        )
        model_gdn2 = DDP(model_gdn2, **ddp_kwargs)

    history_gdn2 = train_model(
        model_gdn2,
        train_dl_gdn2,
        val_dl_gdn2,
        train_cfg,
        device,
        dtype,
        model_name="GDN-2",
    )
    plot_single_model_metrics(
        history_gdn2,
        title="Multi-Head Pure Recurrent GDN-2 (1.3B) Metrics",
        save_path="metrics_gdn2.png",
    )

    # Clean VRAM / memory before starting GatedSurpriseNet
    del model_gdn2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Train GatedSurpriseNet Model
    if IS_MAIN:
        print("\n--- Training Model 2/2: Precision-Weighted GatedSurpriseNet (1.3B Scale) ---")

    train_sampler_surp = DistributedSampler(train_ds, shuffle=True) if WORLD_SIZE > 1 else None
    val_sampler_surp = DistributedSampler(val_ds, shuffle=False) if WORLD_SIZE > 1 else None

    train_dl_surp = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler_surp,
        shuffle=(train_sampler_surp is None),
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    val_dl_surp = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        sampler=val_sampler_surp,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    cfg_surp = Config.from_name(
        "1B_mha",
        block_size=max(data_cfg.max_seq_len, 2048),
        vocab_size=tokenizer.vocab_size,
        padded_vocab_size=tokenizer.vocab_size,
        mixer_type="surprise",
        surprise_net_per_layer=1,
        train_chunk_size=128,
        use_short_conv=True,
        norm_eps=1e-6,
        gradient_checkpointing=True,
        max_write_bound=1.00,
        max_erase_bound=2.00,
        max_precision_bound=2.00,
    )
    model_surp = GPT(cfg_surp).to(device=device, dtype=dtype)
    model_surp.gradient_checkpointing_enable()

    if WORLD_SIZE > 1:
        ddp_kwargs = (
            {"device_ids": [LOCAL_RANK], "output_device": LOCAL_RANK}
            if has_cuda
            else {}
        )
        model_surp = DDP(model_surp, **ddp_kwargs)

    history_surprise = train_model(
        model_surp,
        train_dl_surp,
        val_dl_surp,
        train_cfg,
        device,
        dtype,
        model_name="GatedSurpriseNet",
    )
    plot_single_model_metrics(
        history_surprise,
        title="Multi-Head Pure Recurrent GatedSurpriseNet (1.3B) Metrics",
        save_path="metrics_gated_surprise_net.png",
    )

    # 3. Comparative plot overlaying both curves
    plot_comparison_metrics(
        history_gdn2,
        history_surprise,
        save_path="metrics_comparison_gdn2_vs_surprise.png",
    )

    if WORLD_SIZE > 1:
        dist.barrier()

    if IS_MAIN:
        print("\n" + "=" * 70)
        print("EXPLICIT ARCHITECTURAL COMPARISON SUMMARY")
        print("=" * 70)
        gdn2_train = history_gdn2["train_loss"][-1] if history_gdn2["train_loss"] else float("nan")
        gdn2_ppl = history_gdn2["val_perplexity"][-1] if history_gdn2["val_perplexity"] else float("nan")
        gdn2_nll = history_gdn2["val_nll"][-1] if history_gdn2["val_nll"] else float("nan")
        gdn2_tps = history_gdn2["val_tps"][-1] if history_gdn2["val_tps"] else float("nan")

        surp_train = history_surprise["train_loss"][-1] if history_surprise["train_loss"] else float("nan")
        surp_ppl = history_surprise["val_perplexity"][-1] if history_surprise["val_perplexity"] else float("nan")
        surp_nll = history_surprise["val_nll"][-1] if history_surprise["val_nll"] else float("nan")
        surp_tps = history_surprise["val_tps"][-1] if history_surprise["val_tps"] else float("nan")

        print("  Metric              | GDN-2 Baseline       | GatedSurpriseNet     | Delta")
        print("  " + "-" * 65)
        print(f"  Train Loss          | {gdn2_train:18.4f} | {surp_train:20.4f} | {surp_train - gdn2_train:+.4f}")
        print(f"  Val Perplexity      | {gdn2_ppl:18.2f} | {surp_ppl:20.2f} | {surp_ppl - gdn2_ppl:+.2f}")
        print(f"  Val NLL             | {gdn2_nll:18.2f} | {surp_nll:20.2f} | {surp_nll - gdn2_nll:+.2f}")
        print(f"  Val Throughput      | {gdn2_tps:15.0f} t/s | {surp_tps:17.0f} t/s | {surp_tps - gdn2_tps:+7.0f} t/s")
        print("=" * 70)
        print("[summary] Comparative training complete. Metrics plots saved:")
        print("          1. metrics_gdn2.png")
        print("          2. metrics_gated_surprise_net.png")
        print("          3. metrics_comparison_gdn2_vs_surprise.png")

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
