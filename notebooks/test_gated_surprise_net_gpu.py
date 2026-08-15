"""Kaggle 2xT4 GPU smoke-test + FineWeb-Edu Micro training for Pure Recurrent GatedSurpriseNetAdam Model.

Architecture:
  Pure Recurrent Transformer-like Decoder model (1.3B scale specification) with
  pre-layer RMSNorm, SwiGLU MLP, and GatedSurpriseNetAdam token mixer across ALL layers
  (0 Self-Attention layers).

Part 1 — Sanity checks & 1.3B Architecture Inspection
  1. 1.3B Model Specification & parameter breakdown inspection (100% Recurrent Blocks).
  2. Synthetic overfit on pure recurrent GPT model (all layers GatedSurpriseNetAdam).
  3. Serial-vs-chunk parity: SurpriseMemoryAdam.serial_scan and
     chunk_parallel_training_scan match within tight tolerance.

Part 2 — FineWeb-Edu Micro LM training
  Load bhavnicksm/fineweb-edu-micro (1M tokens, high-quality educational passages),
  build the 1.3B pure recurrent GatedSurpriseNet model, train with
  8-bit AdamW + gradient checkpointing + AMP + DDP + early stopping, and log
  cross-entropy loss, perplexity, and negative log-likelihood.

Designed for Kaggle with 2x T4 GPUs (16GB each). Uses ``torch.distributed`` DDP
when multiple GPUs are detected; falls back to single-GPU/CPU otherwise.

Run on Kaggle::

    python notebooks/test_gated_surprise_net_gpu.py
"""

from __future__ import annotations

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import importlib.metadata
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import fcntl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from datasets import load_dataset
from matplotlib import ticker
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from transformers import AutoTokenizer

REPO_URL = "https://github.com/Gabz4200/QwenDopamine.git"
PIP_REPO_URL = "git+" + REPO_URL

_LOCK_PATH = os.path.join(tempfile.gettempdir(), "qwendopamine_setup.lock")


def _ensure_dependencies() -> None:
    def _check() -> tuple[bool, bool, bool]:
        need_mask = False
        try:
            import transformers.masking_utils

            need_mask = not hasattr(
                transformers.masking_utils, "create_recurrent_attention_mask"
            )
        except (ImportError, AttributeError):
            need_mask = True

        try:
            importlib.metadata.version("qwendopamine")
            need_qwen = False
        except importlib.metadata.PackageNotFoundError:
            need_qwen = True

        need_bnb = False
        if torch.cuda.is_available():
            try:
                import bitsandbytes

                need_bnb = False
            except ImportError:
                need_bnb = True

        return need_mask, need_qwen, need_bnb

    need_mask, need_qwen, need_bnb = _check()
    if not (need_mask or need_qwen or need_bnb):
        return

    with open(_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            need_mask, need_qwen, need_bnb = _check()
            if not (need_mask or need_qwen or need_bnb):
                return

            to_install: list[str] = []
            if need_mask:
                to_install.append("transformers>=4.49.0")
            if need_bnb:
                to_install.append("bitsandbytes>=0.41.0")
            if need_qwen:
                to_install.append(PIP_REPO_URL)

            print(
                "[setup] Installing/upgrading dependencies "
                f"({', '.join(to_install)})..."
            )
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
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


_ensure_dependencies()

from qwendopamine.models.gated_surprise_net import (
    GatedSurpriseNetAdam,
    SurpriseMemoryAdam,
)
from qwendopamine.models.surprise_gpt import (
    Block,
    CausalSelfAttention,
    LLaMAMLP,
    RMSNorm,
    SurpriseGPT,
    SurpriseGPTConfig,
    SwiGLU,
    apply_rotary_emb,
    build_rope_cache,
    compute_model_params,
)

GPT = SurpriseGPT
Config = SurpriseGPTConfig

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

has_cuda = torch.cuda.is_available()
if has_cuda:
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
    print(f"[env] GPU: {torch.cuda.get_device_name(LOCAL_RANK)}")

IS_MAIN = RANK == 0


def inspect_1b_architecture() -> None:
    """Inspect and print the 1.3B scale pure recurrent model specification."""
    cfg_1b = Config.from_name("1B", mixer_type="surprise", surprise_net_per_layer=1)
    stats = compute_model_params(cfg_1b)

    print("=" * 60)
    print(f"1.3B Pure Recurrent Model Specification ({cfg_1b.name} - GatedSurpriseNetAdam)")
    print("=" * 60)
    print(
        f"  Total Parameters:        {stats['total']:,} (~{stats['total']/1e9:.2f}B)"
    )
    print(f"  Layers (n_layer):        {cfg_1b.n_layer}")
    print(f"  Hidden Dim (n_embd):     {cfg_1b.n_embd}")
    print(
        f"  Heads (n_head):          {cfg_1b.n_head} (head_size={cfg_1b.head_size})"
    )
    print(f"  SwiGLU Intermediate:     {cfg_1b.intermediate_size}")
    print(f"  Vocabulary Size:         {cfg_1b.vocab_size:,}")
    print(f"  Max Context Length:      {cfg_1b.block_size:,}")
    print(
        f"  Recurrent Layers:        {stats['num_surprise_layers']}x GatedSurpriseNetAdam (All layers)"
    )
    print(
        f"  Self-Attention Layers:   {stats['num_standard_layers']} (Zero self-attention)"
    )
    print(f"  Recurrent Block Params:  {stats['surprise_block']:,}")
    print(f"  Token Embeddings:        {stats['embed']:,}")
    print("=" * 60)


def run_synthetic_overfit() -> tuple[float, float, bool]:
    """Verify learning on a small pure recurrent GPT model (all layers GatedSurpriseNetAdam)."""
    torch.manual_seed(0)
    vocab_size = 64
    seq_len = 32
    batch_size = 4
    synth_steps = 200

    cfg = Config(
        name="synthetic_recurrent_surprise",
        block_size=seq_len,
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        head_size=32,
        n_query_groups=4,
        intermediate_size=344,
        norm_eps=1e-5,
        train_chunk_size=seq_len,
        surprise_net_per_layer=1,
        mixer_type="surprise",
    )
    model = GPT(cfg).to(device=device, dtype=dtype)

    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if IS_MAIN:
        print(f"[synth] params: {total_params:,}  trainable: {trainable:,}")
        print(f"[synth] All {cfg.n_layer} layers use GatedSurpriseNetAdam (0 self-attention)")

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
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.item())
        final_loss_val = float(loss.item())
        if step % 50 == 0 and IS_MAIN:
            print(f"[synth] step {step:03d}  loss={loss.item():.4f}")

    if IS_MAIN:
        print(f"[synth] initial={initial_loss:.4f}  final={final_loss_val:.4f}")

    passed = final_loss_val < initial_loss * 0.5 and math.isfinite(
        final_loss_val
    )
    return initial_loss, final_loss_val, passed


@torch.no_grad()
def run_serial_chunk_parity() -> bool:
    """Verify parity between serial scan and chunk parallel scan."""
    bs, t, h, d = 2, 32, 4, 16
    torch.manual_seed(42)
    q = torch.randn(bs, t, h, d, device=device, dtype=dtype)
    k = torch.randn(bs, t, h, d, device=device, dtype=dtype)
    v = torch.randn(bs, t, h, d, device=device, dtype=dtype)
    g = torch.randn(bs, t, h, d, device=device, dtype=dtype).abs_().mul_(-1)
    b = torch.rand(bs, t, h, d, device=device, dtype=dtype)
    w = torch.rand(bs, t, h, d, device=device, dtype=dtype)

    memory = SurpriseMemoryAdam(num_heads=h, head_k_dim=d, head_v_dim=d).to(
        device=device, dtype=dtype
    )

    out_s, _, nll_s = memory.serial_scan(q, k, v, g, b, w)
    out_c, _, nll_c = memory.chunk_parallel_training_scan(
        q, k, v, g, b, w, chunk_size=16
    )

    passed = torch.allclose(out_s, out_c, atol=1e-3) and torch.allclose(
        nll_s, nll_c, atol=1e-3
    )
    if not passed and IS_MAIN:
        print("[parity] serial vs chunk: FAIL")
        print(f"  output close: {torch.allclose(out_s, out_c, atol=1e-3)}")
        print(f"  nll close: {torch.allclose(nll_s, nll_c, atol=1e-3)}")
    return passed


@dataclass
class FineWebMicroConfig:
    dataset_name: str = "bhavnicksm/fineweb-edu-micro"
    max_seq_len: int = 512
    max_train_examples: int | None = None
    max_val_examples: int | None = None
    batch_size: int = 1
    val_split_ratio: float = 0.1
    seed: int = 42


def load_fineweb_micro_tokenized(
    cfg: FineWebMicroConfig, tokenizer: Any
) -> tuple[TensorDataset, TensorDataset]:
    """Load and tokenize passages from bhavnicksm/fineweb-edu-micro."""
    if IS_MAIN:
        print(f"[data] Loading dataset '{cfg.dataset_name}' ...")
    try:
        ds = load_dataset(cfg.dataset_name)
    except TypeError:
        ds = load_dataset(cfg.dataset_name, trust_remote_code=True)

    raw_train = ds["train"]
    split = raw_train.train_test_split(
        test_size=cfg.val_split_ratio, seed=cfg.seed
    )

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
    log_interval: int = 25
    eval_interval: int = 100
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 50
    grad_clip: float = 1.0
    grad_accum_steps: int = 8
    use_amp: bool = True
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-3


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
    nll_total = 0.0
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
            nll_total += float(loss.item()) * batch_tokens

    eval_time = time.perf_counter() - eval_t0

    if WORLD_SIZE > 1:
        stats = torch.tensor(
            [total_loss, total_tokens, nll_total], device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_tokens, nll_total = stats.tolist()

    avg_loss = total_loss / max(total_tokens, 1.0)
    perplexity = math.exp(min(avg_loss, 50))
    tps = total_tokens / max(eval_time, 1e-9)
    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
        "val_nll": nll_total,
        "val_tokens": total_tokens,
        "val_tps": tps,
    }


def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, list[float]]:
    model.train()

    optimizer_cls = torch.optim.AdamW
    if device.type == "cuda":
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.PagedAdamW8bit
            if IS_MAIN:
                print("[opt] Using bitsandbytes PagedAdamW8bit (low memory footprint for 1.3B)")
        except ImportError:
            if IS_MAIN:
                print("[opt] bitsandbytes not found, falling back to standard AdamW")

    optimizer = optimizer_cls(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        eps=1e-8,
    )
    loss_fn = nn.CrossEntropyLoss()

    try:
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=(
                device.type == "cuda"
                and dtype == torch.float16
                and cfg.use_amp
            ),
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=(
                device.type == "cuda"
                and dtype == torch.float16
                and cfg.use_amp
            )
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

    best_val_loss = float("inf")
    patience_counter = 0

    while step < cfg.num_steps:
        if isinstance(train_dl.sampler, DistributedSampler):
            train_dl.sampler.set_epoch(epoch)
        epoch += 1

        for xb, yb in train_dl:
            if step >= cfg.num_steps:
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
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)
                )
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += float(loss.item()) * cfg.grad_accum_steps
            tokens_seen += yb.numel()

            if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == cfg.num_steps:
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step_time = time.perf_counter() - t0

            if step % cfg.log_interval == 0 and IS_MAIN:
                avg_train_loss = accum_loss / max(
                    1, (step % cfg.grad_accum_steps) + 1
                )
                history["train_loss"].append(avg_train_loss)
                history["train_steps"].append(float(step))
                history["tokens_seen"].append(float(tokens_seen))
                history["lr"].append(lr)
                history["step_time_s"].append(step_time)
                print(
                    f"[train] step {step:5d}  loss={avg_train_loss:.4f}  "
                    f"lr={lr:.2e}  tok/s={tokens_seen / max(step_time, 1e-9):.0f}"
                )
                accum_loss = 0.0

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
                        f"[eval]  step {step:5d}  loss={val_loss:.4f}  "
                        f"ppl={metrics['val_perplexity']:.2f}  "
                        f"nll={metrics['val_nll']:.2f}  "
                        f"tps={metrics['val_tps']:.0f}"
                    )

                    # Early stopping check
                    if val_loss < best_val_loss - cfg.early_stopping_min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= cfg.early_stopping_patience:
                            print(
                                f"[early_stop] Validation loss plateaued for {patience_counter} "
                                f"evaluations (best={best_val_loss:.4f}). Stopping early."
                            )
                            break
                model.train()

            step += 1

        if patience_counter >= cfg.early_stopping_patience:
            break

    metrics = evaluate(model, val_dl, loss_fn, device, dtype)
    if IS_MAIN:
        history["val_loss"].append(metrics["val_loss"])
        history["val_perplexity"].append(metrics["val_perplexity"])
        history["val_nll"].append(metrics["val_nll"])
        history["val_tps"].append(metrics["val_tps"])
        history["val_steps"].append(float(step))
        print(
            f"[final] loss={metrics['val_loss']:.4f}  "
            f"ppl={metrics['val_perplexity']:.2f}  "
            f"nll={metrics['val_nll']:.2f}  "
            f"tps={metrics['val_tps']:.0f}"
        )
    return history


def plot_metrics(
    history: dict[str, list[float]], save_path: str = "metrics.png"
) -> None:
    if not IS_MAIN:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Pure Recurrent GatedSurpriseNetAdam (1.3B) — FineWeb-Edu Micro Training Metrics",
        fontsize=13,
    )

    plots = [
        (
            axes[0, 0],
            "train_steps",
            "train_loss",
            "Train Loss",
            "Loss",
            "tab:blue",
        ),
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

    for ax, x_key, y_key, title, ylabel, color in plots:
        x_data = history.get(x_key, [])
        y_data = history.get(y_key, [])
        if x_data and y_data and len(x_data) == len(y_data):
            ax.plot(x_data, y_data, marker="o", color=color)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.grid(True)

    axes[1, 2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[plot] Saved metrics plot to {save_path}")
    plt.close(fig)


def main() -> None:
    if IS_MAIN:
        inspect_1b_architecture()

    sanity_ok = True

    if IS_MAIN:
        print("\n===== Sanity check 1: synthetic overfit =====")
    initial_loss, final_loss, synth_ok = run_synthetic_overfit()
    if not synth_ok:
        sanity_ok = False
        if IS_MAIN:
            print(
                f"[check] Synthetic overfit: FAIL — loss {initial_loss:.4f} -> {final_loss:.4f}"
            )
    elif IS_MAIN:
        print("[check] Synthetic overfit: PASS")

    if IS_MAIN:
        print("\n===== Sanity check 2: serial vs chunk parity =====")
    parity_ok = run_serial_chunk_parity()
    if not parity_ok:
        sanity_ok = False
        if IS_MAIN:
            print("[check] Serial vs chunk parity: FAIL")
    elif IS_MAIN:
        print("[check] Serial vs chunk parity: PASS")

    if WORLD_SIZE > 1:
        sanity_tensor = torch.tensor([1 if sanity_ok else 0], device=device)
        dist.broadcast(sanity_tensor, src=0)
        sanity_ok = bool(sanity_tensor.item())

    if IS_MAIN:
        if sanity_ok:
            print(
                "[check] All sanity checks passed. Proceeding to FineWeb-Edu Micro training."
            )
        else:
            print(
                "[check] One or more sanity checks failed. Skipping FineWeb-Edu Micro training."
            )

    if not sanity_ok:
        if WORLD_SIZE > 1:
            dist.destroy_process_group()
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if IS_MAIN:
        print("\n===== Part 2: FineWeb-Edu Micro 1.3B LM training =====")

    data_cfg = FineWebMicroConfig(
        max_seq_len=512,
        batch_size=1,
        val_split_ratio=0.1,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 1_000_000
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_fineweb_micro_tokenized(data_cfg, tokenizer)

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True) if WORLD_SIZE > 1 else None
    )
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False) if WORLD_SIZE > 1 else None
    )

    pin_memory = device.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
    )

    train_cfg = TrainConfig(
        num_steps=1000,
        log_interval=25,
        eval_interval=100,
        lr=3e-4,
        min_lr=1e-5,
        weight_decay=0.1,
        warmup_steps=50,
        grad_accum_steps=8,
        early_stopping_patience=4,
        early_stopping_min_delta=1e-3,
    )

    lm_cfg = Config.from_name(
        "1B",
        block_size=max(data_cfg.max_seq_len, 2048),
        vocab_size=tokenizer.vocab_size,
        padded_vocab_size=tokenizer.vocab_size,
        mixer_type="surprise",
        surprise_net_per_layer=1,
        gradient_checkpointing=True,
    )

    model = GPT(lm_cfg).to(device=device, dtype=dtype)
    model.gradient_checkpointing_enable()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    total_params = sum(p.numel() for p in model.parameters())
    if IS_MAIN:
        print(
            f"[model] Pure Recurrent 1.3B LM  params={total_params:,}  "
            f"layers={lm_cfg.n_layer} (100% GatedSurpriseNetAdam, 0 self-attention)"
        )

    history = train(model, train_dl, val_dl, train_cfg, device, dtype)

    plot_metrics(history, save_path="metrics.png")

    if WORLD_SIZE > 1:
        dist.barrier()

    if IS_MAIN:
        print("\n===== Summary =====")
        final_train = (
            history["train_loss"][-1]
            if history["train_loss"]
            else float("nan")
        )
        final_ppl = (
            history["val_perplexity"][-1]
            if history["val_perplexity"]
            else float("nan")
        )
        final_nll = (
            history["val_nll"][-1] if history["val_nll"] else float("nan")
        )
        final_tps = (
            history["val_tps"][-1] if history["val_tps"] else float("nan")
        )
        print(f"  Train loss (last logged): {final_train:.4f}")
        print(f"  Val perplexity:          {final_ppl:.2f}")
        print(f"  Val NLL:                 {final_nll:.2f}")
        print(f"  Val throughput:          {final_tps:.0f} tok/s")
        print(f"  Params:                  {total_params:,}")
        print(f"  GPUs:                    {WORLD_SIZE}x T4")
        print("  Data:                    bhavnicksm/fineweb-edu-micro")
        print(
            f"  Architecture:            Pure Recurrent (1.3B, Layers: {lm_cfg.n_layer}, Mixer: GatedSurpriseNetAdam, 0 Self-Attention)"
        )
        print(f"  Train steps:             {train_cfg.num_steps}")
        print("  Result: Pure Recurrent GatedSurpriseNetAdam 1.3B LM training complete.")

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
