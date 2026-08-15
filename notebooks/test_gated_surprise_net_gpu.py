"""Kaggle 2xT4 GPU smoke-test + WikiText-2 training for GatedSurpriseNetAdam.

Part 1 — Sanity checks
  1. Tiny synthetic overfit: one batch, few steps, loss must drop.
  2. Serial-vs-chunk parity: SurpriseMemoryAdam.serial_scan and
     chunk_parallel_training_scan must match within tight tolerance.

Part 2 — WikiText-2 LM training
  Load Salesforce/wikitext, build a small 2-layer decoder whose only
  mixer is GatedSurpriseNetAdam, train for a fixed number of steps,
  and log per-token cross-entropy loss, perplexity, and negative
  log-likelihood every ``log_interval`` steps.

Designed for Kaggle with 2x T4 GPUs. Uses ``torch.distributed`` DDP
when multiple GPUs are detected; falls back to single-GPU otherwise.

Run on Kaggle::

    python notebooks/test_gated_surprise_net_gpu.py

Dependencies: the ``gpu`` extra in ``pyproject.toml`` (``flash-linear-attention``
is optional here because GatedSurpriseNetAdam runs in pure-PyTorch mode).
"""

from __future__ import annotations

import importlib.metadata
import math
import os
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
    def _check() -> tuple[bool, bool]:
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

        return need_mask, need_qwen

    need_mask, need_qwen = _check()
    if not (need_mask or need_qwen):
        return

    # Kaggle torchrun launches multiple ranks in the same container;
    # serialize pip installs so two ranks don't race.
    with open(_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            need_mask, need_qwen = _check()
            if not (need_mask or need_qwen):
                return

            to_install: list[str] = []
            if need_mask:
                to_install.append("transformers>=4.49.0")
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

from qwendopamine.models.gated_surprise_net import SurpriseMemoryAdam
from qwendopamine.models.model_factory import ResearchDecoder

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

print(f"[env] rank={RANK}/{WORLD_SIZE}  local_rank={LOCAL_RANK}  device={device}  dtype={dtype}")
if has_cuda:
    print(f"[env] GPU: {torch.cuda.get_device_name(LOCAL_RANK)}")

IS_MAIN = RANK == 0

HIDDEN_SIZE = 256
NUM_HEADS = 4
HEAD_DIM = 64
VOCAB_SIZE = 64
SEQ_LEN = 32
BATCH_SIZE = 4
SYNTH_STEPS = 200


class TinySurpriseConfig:
    hidden_size: int = HIDDEN_SIZE
    vocab_size: int = VOCAB_SIZE
    num_heads: int = NUM_HEADS
    head_dim: int = HEAD_DIM
    num_layers: int = 2
    max_position_embeddings: int = 2048
    block_types: list[str] = ["gated_surprise_net", "gated_surprise_net"]
    rms_norm_eps: float = 1e-5
    use_short_conv: bool = True
    conv_size: int = 4
    expand_v: float = 1.0
    train_chunk_size: int = 128


def build_tiny_model(cfg: TinySurpriseConfig) -> nn.Module:
    return ResearchDecoder(cfg)


def run_synthetic_overfit() -> tuple[float, float, bool]:
    torch.manual_seed(0)
    cfg = TinySurpriseConfig()
    cfg.vocab_size = VOCAB_SIZE
    cfg.max_position_embeddings = SEQ_LEN
    cfg.train_chunk_size = SEQ_LEN
    model = build_tiny_model(cfg).to(device=device, dtype=dtype)

    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if IS_MAIN:
        print(f"[synth] params: {total_params:,}  trainable: {trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    initial_loss = 0.0
    final_loss_val = 0.0
    for step in range(SYNTH_STEPS):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.item())
        final_loss_val = float(loss.item())
        if step % 50 == 0 and IS_MAIN:
            print(f"[synth] step {step:03d}  loss={loss.item():.4f}")

    if IS_MAIN:
        print(f"[synth] initial={initial_loss:.4f}  final={final_loss_val:.4f}")

    passed = final_loss_val < initial_loss * 0.5 and math.isfinite(final_loss_val)
    return initial_loss, final_loss_val, passed


@torch.no_grad()
def run_serial_chunk_parity() -> bool:
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

    passed = torch.allclose(out_s, out_c, atol=1e-3) and torch.allclose(nll_s, nll_c, atol=1e-3)
    if not passed and IS_MAIN:
        print("[parity] serial vs chunk: FAIL")
        print(f"  output close: {torch.allclose(out_s, out_c, atol=1e-3)}")
        print(f"  nll close: {torch.allclose(nll_s, nll_c, atol=1e-3)}")
    return passed


@dataclass
class WikiTextConfig:
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_seq_len: int = 128
    max_train_examples: int | None = 5000
    max_val_examples: int | None = 1000
    batch_size: int = 16


def load_wikitext_tokenized(
    cfg: WikiTextConfig, tokenizer: Any
) -> tuple[TensorDataset, TensorDataset]:
    if IS_MAIN:
        print(f"[data] Loading {cfg.dataset_name} ({cfg.dataset_config}) ...")
    try:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    except TypeError:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config, trust_remote_code=True)

    def encode_split(split: str, max_examples: int | None) -> list[list[int]]:
        texts: list[str] = []
        for ex in ds[split]:
            txt = ex.get("text", "").strip()
            if txt:
                texts.append(txt)
            if max_examples is not None and len(texts) >= max_examples:
                break
        if IS_MAIN:
            print(f"[data] {split}: {len(texts)} passages")
        seqs: list[list[int]] = []
        for txt in texts:
            ids = tokenizer(txt, truncation=False, add_special_tokens=False)[
                "input_ids"
            ]
            if len(ids) < 2:
                continue
            for i in range(0, len(ids) - cfg.max_seq_len, cfg.max_seq_len):
                window = ids[i : i + cfg.max_seq_len + 1]
                if len(window) == cfg.max_seq_len + 1:
                    seqs.append(window)
        return seqs

    train_seqs = encode_split("train", cfg.max_train_examples)
    val_seqs = encode_split("validation", cfg.max_val_examples)

    if not train_seqs:
        raise RuntimeError("No training sequences produced.")

    def to_tensor(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.tensor([s[:-1] for s in seqs], dtype=torch.long)
        targets = torch.tensor([s[1:] for s in seqs], dtype=torch.long)
        return inputs, targets

    train_in, train_tgt = to_tensor(train_seqs)
    val_in, val_tgt = to_tensor(val_seqs)
    if IS_MAIN:
        print(f"[data] train sequences: {train_in.shape[0]}  val sequences: {val_in.shape[0]}")
    return TensorDataset(train_in, train_tgt), TensorDataset(val_in, val_tgt)


@dataclass
class TrainConfig:
    num_steps: int = 1000
    log_interval: int = 50
    eval_interval: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    use_amp: bool = True
    chunk_size: int = 128


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
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                logits = model(xb)
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)
            )
            batch_tokens = yb.numel()
            total_loss += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens
            nll_total += float(loss.item()) * batch_tokens

    eval_time = time.perf_counter() - eval_t0

    if WORLD_SIZE > 1:
        stats = torch.tensor([total_loss, total_tokens, nll_total], device=device)
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
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
                lr = cfg.lr * (0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda"),
            ):
                logits = model(xb)
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tokens_seen += yb.numel()
            step_time = time.perf_counter() - t0

            if step % cfg.log_interval == 0 and IS_MAIN:
                history["train_loss"].append(float(loss.item()))
                history["train_steps"].append(float(step))
                history["tokens_seen"].append(float(tokens_seen))
                history["lr"].append(lr)
                history["step_time_s"].append(step_time)
                print(
                    f"[train] step {step:5d}  loss={loss.item():.4f}  "
                    f"lr={lr:.2e}  tok/s={tokens_seen / max(step_time, 1e-9):.0f}"
                )

            if step % cfg.eval_interval == 0 and step > 0:
                metrics = evaluate(
                    model, val_dl, loss_fn, device, dtype, max_batches=50
                )
                if IS_MAIN:
                    history["val_loss"].append(metrics["val_loss"])
                    history["val_perplexity"].append(metrics["val_perplexity"])
                    history["val_nll"].append(metrics["val_nll"])
                    history["val_tps"].append(metrics["val_tps"])
                    history["val_steps"].append(float(step))
                    print(
                        f"[eval]  step {step:5d}  loss={metrics['val_loss']:.4f}  "
                        f"ppl={metrics['val_perplexity']:.2f}  "
                        f"nll={metrics['val_nll']:.2f}  "
                        f"tps={metrics['val_tps']:.0f}"
                    )
                model.train()

            step += 1

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


def plot_metrics(history: dict[str, list[float]], save_path: str = "metrics.png") -> None:
    if not IS_MAIN:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("GatedSurpriseNetAdam — WikiText-2 Training Metrics", fontsize=13)

    plots = [
        (axes[0, 0], "train_steps", "train_loss", "Train Loss", "Loss", "tab:blue"),
        (axes[0, 1], "val_steps", "val_loss", "Val Loss", "Loss", "tab:orange"),
        (axes[0, 2], "val_steps", "val_perplexity", "Val Perplexity", "Perplexity", "tab:green"),
        (axes[1, 0], "val_steps", "val_nll", "Val NLL", "NLL", "tab:red"),
        (axes[1, 1], "val_steps", "val_tps", "Val Throughput", "Tokens / sec", "tab:purple"),
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
    sanity_ok = True

    if IS_MAIN:
        print("\n===== Sanity check 1: synthetic overfit =====")
    initial_loss, final_loss, synth_ok = run_synthetic_overfit()
    if not synth_ok:
        sanity_ok = False
        if IS_MAIN:
            print(f"[check] Synthetic overfit: FAIL — loss {initial_loss:.4f} -> {final_loss:.4f}")
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
            print("[check] All sanity checks passed. Proceeding to WikiText-2 training.")
        else:
            print("[check] One or more sanity checks failed. Skipping WikiText-2 training.")

    if not sanity_ok:
        if WORLD_SIZE > 1:
            dist.destroy_process_group()
        return

    if IS_MAIN:
        print("\n===== Part 2: WikiText-2 training =====")

    wt_cfg = WikiTextConfig(
        max_seq_len=128,
        max_train_examples=5000,
        max_val_examples=1000,
        batch_size=16,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_wikitext_tokenized(wt_cfg, tokenizer)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if WORLD_SIZE > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if WORLD_SIZE > 1 else None

    pin_memory = (device.type == "cuda")
    train_dl = DataLoader(
        train_ds,
        batch_size=wt_cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=wt_cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=pin_memory,
    )

    train_cfg = TrainConfig(
        num_steps=1000,
        log_interval=50,
        eval_interval=200,
        lr=3e-4,
        weight_decay=0.1,
        warmup_steps=100,
        chunk_size=128,
    )

    lm_cfg = TinySurpriseConfig()
    lm_cfg.vocab_size = tokenizer.vocab_size
    lm_cfg.hidden_size = HIDDEN_SIZE
    lm_cfg.num_heads = NUM_HEADS
    lm_cfg.head_dim = HEAD_DIM
    lm_cfg.num_layers = 2
    lm_cfg.block_types = ["gated_surprise_net"] * 2
    lm_cfg.max_position_embeddings = max(wt_cfg.max_seq_len, 2048)
    lm_cfg.train_chunk_size = train_cfg.chunk_size

    model = build_tiny_model(lm_cfg).to(device=device, dtype=dtype)
    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    total_params = sum(p.numel() for p in model.parameters())
    if IS_MAIN:
        print(f"[model] GatedSurpriseNetAdam LM  params={total_params:,}")

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
        print("  Data:                    Salesforce/wikitext-2-raw-v1")
        print(f"  Block:                   GatedSurpriseNetAdam x {lm_cfg.num_layers}")
        print(f"  Train steps:             {train_cfg.num_steps}")
        print("  Result: GatedSurpriseNetAdam LM training complete.")

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

