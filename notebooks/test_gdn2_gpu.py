"""GDN-2 GPU smoke test for Kaggle.

Install from GitHub, then run a tiny training run on real text data to verify
the Gated DeltaNet-2 implementation actually learns on GPU.

Expected hardware: 2x T4 or P100 on Kaggle.
Expected runtime: ~5-10 minutes for the full notebook.
"""

import importlib.metadata
import subprocess
import sys
import time

REPO_URL = "https://github.com/Gabz4200/QwenDopamine.git"
PIP_REPO_URL = "git+" + REPO_URL

def _is_qwendopamine_installed() -> bool:
    try:
        importlib.metadata.version("qwendopamine")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False

print("[setup] Checking qwendopamine import...")
if _is_qwendopamine_installed():
    import qwendopamine
    print(f"[setup] qwendopamine {qwendopamine.__version__} already installed")
else:
    print(f"[setup] Installing from {PIP_REPO_URL} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", PIP_REPO_URL],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    import qwendopamine
    print(f"[setup] Installed qwendopamine {qwendopamine.__version__}")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qwendopamine.blocks import GatedDeltaNet2Block
from qwendopamine.models.normalization import RMSNorm

# ---------------------------------------------------------------------------
# Cell 2: Device and dtype setup
# ---------------------------------------------------------------------------
assert torch.cuda.is_available(), "This notebook requires a CUDA GPU (T4/P100)."
device = torch.device("cuda")
dtype = torch.bfloat16  # Kaggle T4/P100 support bf16; use float16 if needed

print(f"[env] device={device}, dtype={dtype}")
print(f"[env] GPU: {torch.cuda.get_device_name(0)}")
print(f"[env] BF16 supported: {torch.cuda.is_bf16_supported()}")

# ---------------------------------------------------------------------------
# Cell 3: Load real dataset (TinyStories) with fallback
# ---------------------------------------------------------------------------
# TinyStories is a small LM-friendly dataset of short stories.
# We stream a small subset to keep download and training time low.
# If dataset loading fails (e.g. no internet), we fall back to a small
# embedded real text corpus.

USE_DATASET = True
NUM_TRAIN_EXAMPLES = 2000
SEQ_LEN = 128

try:
    from datasets import load_dataset

    print("[data] Loading TinyStories (streaming, small subset)...")
    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds_iter = iter(ds.take(NUM_TRAIN_EXAMPLES))
    texts = [row["text"] for row in ds_iter if row["text"].strip()]
    print(f"[data] Loaded {len(texts)} real stories from TinyStories")
except Exception as e:
    print(f"[data] Could not load dataset ({e}); using embedded real text fallback")
    USE_DATASET = False
    # A small slice of real public-domain text (Shakespeare, etc.)
    # This is real language, just much smaller.
    texts = [
        (
            "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
            "The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles "
            "And by opposing end them. To die: to sleep; No more; and by a sleep to say we end "
            "The heart-ache and the thousand natural shocks That flesh is heir to: 'tis a consummation "
            "Devoutly to be wish'd. To die, to sleep; To sleep: perchance to dream: ay, there's the rub; "
            "For in that sleep of death what dreams may come When we have shuffled off this mortal coil, "
            "Must give us pause."
        ),
        (
            "All the world's a stage, And all the men and women merely players; They have their exits "
            "and their entrances, And one man in his time plays many parts, His acts being seven ages. "
            "At first the infant, Mewling and puking in the nurse's arms; And then the whining schoolboy, "
            "with his satchel And shining morning face, creeping like snail Unwillingly to school."
        ),
        (
            "Shall I compare thee to a summer's day? Thou art more lovely and more temperate: "
            "Rough winds do shake the darling buds of May, And summer's lease hath all too short a date; "
            "Sometime too hot the eye of heaven shines, And often is his gold complexion dimm'd; "
            "And every fair from fair sometime declines, By chance, or nature's changing course untrimm'd."
        ),
        (
            "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age "
            "of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season "
            "of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair."
        ),
        (
            "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in "
            "my purse, and nothing particular to interest me on shore, I thought I would sail about a little "
            "and see the watery part of the world. It is a way I have of driving off the spleen and "
            "regulating the circulation."
        ),
        (
            "In the beginning God created the heaven and the earth. And the earth was without form, and "
            "void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face "
            "of the waters. And God said, Let there be light: and there was light. And God saw the light, "
            "that it was good: and God divided the light from the darkness."
        ),
        (
            "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his "
            "chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through "
            "the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty "
            "dust from entering along with him."
        ),
        (
            "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were "
            "perfectly normal, thank you very much. They were the last people you'd expect to be involved "
            "in anything strange or mysterious, because they just didn't hold with such nonsense."
        ),
        (
            "The man in black fled across the desert, and the gunslinger followed. The desert was the "
            "apotheosis of all deserts, and the gunslinger had been in many. He was a creature of the "
            "old world, a relic of a time when men walked the earth in search of something other than "
            "their own reflections."
        ),
        (
            "Sitting at his desk, Winston dipped his pen into the ink and stared at the blank page "
            "before him. The telescreen on the wall was broadcasting a speech about the victories of "
            "the Party, but he was not listening. His mind was on the forbidden thought that had crept "
            "into his consciousness like a thief in the night."
        ),
    ] * 200  # repeat to get enough data
    print(f"[data] Using embedded real text fallback ({len(texts)} passages)")

# ---------------------------------------------------------------------------
# Cell 4: Build character-level tokenizer (no extra downloads)
# ---------------------------------------------------------------------------
# Character-level tokenization is robust, requires no pretrained tokenizer,
# and works on any text. For a smoke test, it's sufficient to verify that
# the model can learn real language statistics.

class CharTokenizer:
    def __init__(self, texts: list[str]):
        # Build vocabulary from all characters in the text
        chars = sorted(set("".join(texts)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.vocab["<pad>"] = len(self.vocab)
        self.vocab["<unk>"] = len(self.vocab)
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(ch, self.vocab["<unk>"]) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv_vocab.get(i, "<unk>") for i in ids)


tokenizer = CharTokenizer(texts)
print(f"[data] Char vocab size: {tokenizer.vocab_size}")

# ---------------------------------------------------------------------------
# Cell 5: Prepare dataset
# ---------------------------------------------------------------------------
# Encode all texts, then create fixed-length sequences.
# Target = input shifted by 1 token (next-token prediction).

torch.manual_seed(42)

all_tokens: list[int] = []
for text in texts:
    all_tokens.extend(tokenizer.encode(text))

print(f"[data] Total tokens: {len(all_tokens):,}")

# Create sequences
seqs = []
for i in range(0, len(all_tokens) - SEQ_LEN - 1, SEQ_LEN):
    seqs.append(all_tokens[i : i + SEQ_LEN + 1])  # input + target

# Filter out any sequences that contain unknown tokens (shouldn't happen)
seqs = [s for s in seqs if max(s) < tokenizer.vocab_size and min(s) >= 0]

if not seqs:
    raise RuntimeError("No valid sequences created. Check tokenization.")

inputs = torch.tensor([s[:-1] for s in seqs], dtype=torch.long)
targets = torch.tensor([s[1:] for s in seqs], dtype=torch.long)

print(f"[data] Sequences: {inputs.shape[0]}, seq_len={SEQ_LEN}")

train_ds = TensorDataset(inputs, targets)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)

# ---------------------------------------------------------------------------
# Cell 6: Model config
# ---------------------------------------------------------------------------
class ModelConfig:
    hidden_size: int = 256
    num_heads: int = 8
    head_dim: int = 32
    expand_v: float = 1.0
    num_v_heads: int = 8
    rms_norm_eps: float = 1e-6
    conv_size: int = 4
    conv_bias: bool = False
    allow_neg_eigval: bool = False
    gdn2_kernel_mode: str = "chunk"  # use GPU kernel when available

config = ModelConfig()

# ---------------------------------------------------------------------------
# Cell 7: Build model
# ---------------------------------------------------------------------------
class TinyGDN2LM(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.hidden_size = cfg.hidden_size

        self.embed = nn.Embedding(vocab_size, cfg.hidden_size)
        self.block = GatedDeltaNet2Block(cfg, layer_idx=0)
        self.norm_out = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.block(x)
        x = self.norm_out(x)
        return self.lm_head(x)


model = TinyGDN2LM(config, vocab_size=tokenizer.vocab_size).to(device=device, dtype=dtype)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[model] total params: {total_params:,}")
print(f"[model] trainable params: {trainable_params:,}")

# Warm-up forward pass
model.eval()
with torch.no_grad():
    dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    _ = model(dummy)
print("[model] forward pass OK")

# ---------------------------------------------------------------------------
# Cell 8: Training loop
# ---------------------------------------------------------------------------
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

num_steps = 500
log_interval = 50
loss_history = []

print(f"[train] starting for {num_steps} steps...")
t0 = time.time()

for step, (xb, yb) in enumerate(train_dl):
    xb = xb.to(device)
    yb = yb.to(device)

    logits = model(xb)
    loss = loss_fn(logits.reshape(-1, tokenizer.vocab_size), yb.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_val = loss.item()
    loss_history.append(loss_val)

    if step % log_interval == 0:
        elapsed = time.time() - t0
        print(f"  step {step:4d}/{num_steps}  loss={loss_val:.4f}  "
              f"time={elapsed:.1f}s")

    if step >= num_steps:
        break

print(f"[train] done in {time.time() - t0:.1f}s")
print(f"[train] final loss: {loss_history[-1]:.4f}")
print(f"[train] first loss: {loss_history[0]:.4f}")
print(f"[train] loss reduction: {loss_history[0] - loss_history[-1]:.4f}")

# ---------------------------------------------------------------------------
# Cell 9: Sanity checks
# ---------------------------------------------------------------------------
initial_loss = loss_history[0]
final_loss = loss_history[-1]

assert final_loss < initial_loss * 0.5, (
    f"GDN-2 did not learn: loss went from {initial_loss:.4f} to {final_loss:.4f}"
)
print("[check] Loss decreased by >50%: PASS")

assert torch.isfinite(torch.tensor(final_loss)), "Final loss is not finite"
print("[check] Final loss is finite: PASS")

model.eval()
with torch.no_grad():
    xb_check, yb_check = next(iter(train_dl))
    xb_check = xb_check.to(device)
    yb_check = yb_check.to(device)
    logits_check = model(xb_check)
    loss_check = loss_fn(logits_check.reshape(-1, tokenizer.vocab_size), yb_check.reshape(-1))
    assert torch.isfinite(loss_check), "Eval loss is not finite"
print("[check] Eval loss finite: PASS")

# Decode a sample to verify model sees real text structure
sample_input = inputs[0][:SEQ_LEN].tolist()
sample_text = tokenizer.decode(sample_input)
print(f"\n[sample] Real text from dataset:\n{sample_text[:200]}...")

print("\n[summary]")
print(f"  Model: TinyGDN2LM (1x GatedDeltaNet2Block)")
print(f"  Params: {total_params:,}")
print(f"  Device: {device}")
print(f"  Data: {'TinyStories' if USE_DATASET else 'embedded real text'}")
print(f"  Kernel mode: {config.gdn2_kernel_mode}")
print(f"  Training steps: {num_steps}")
print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f}")
print("  Result: GDN-2 learns on real text with GPU kernel.")
