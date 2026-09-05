
## Notebook/Kaggle and PEFT pipeline lessons (2026-09-05)

- Kaggle notebook install cells must avoid `git+https` pip URLs; use GitHub Releases wheel + `importlib.util.find_spec` guard.
- Pin deps to actually-existing versions: `transformers>=4.40.0,<5.0.0`, `torch>=2.0.0`, `unsloth>=4.7` per pyproject extras.
- Root `OUTPUT_DIR` at `/kaggle/working` and append `os.getpid()` to avoid collisions on reruns.
- `InfiniDopamineConfig(**hf_cfg.to_dict())` is the canonical config construction; do not hand-roll defaults via `_as_obj`.
- Verified PEFT pipeline: 27 non-Linear missing keys are direct-trained; 16 LoRA targets resolve to `nn.Linear`; re-unfreeze reward_branch after PEFT wrap; `merge_and_unload()` preserves direct-trained weights and total param count.
- `SFTTrainer` now uses `processing_class=`; `optim="paged_adamw_8bit"` needs `bitsandbytes` fallback.
- Tokenize empty rows as empty lists, then filter; never return None from `IterableDataset.map`.
- Notebook passes all quality gates: ruff, pyrefly 0 errors, jupytext sync, syntax valid.
