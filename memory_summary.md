
## Notebook/Kaggle lessons (2026-09-05)

(16) **Kaggle pip install must avoid VCS URLs.** `%pip install git+https://...` is blocked by default. Use a pre-built wheel from GitHub Releases, and guard with `importlib.util.find_spec("qwendopamine")` so the cell is idempotent across kernel restarts.

(17) **Pin package versions to what actually exists.** `transformers>=5.0.0` does not exist yet; use `>=4.40.0,<5.0.0`. `torch>=2.9.0` is future; use `>=2.0.0`. `unsloth>=2024.7` is GPU-only; per pyproject extras use `unsloth>=4.7`.

(18) **Kaggle output directory.** Always root `OUTPUT_DIR` at `/kaggle/working` (or `KAGGLE_WORKING_DIR` env var) so saved checkpoints persist after the session. Use `os.path.join` and include `os.getpid()` to avoid collisions on reruns.

(19) **PEFT LoRA targets must resolve to `nn.Linear`.** Many plausible module names (`q_proj`, `gate_proj`, `up_proj`, `down_proj`, `merger.linear_fc1`, etc.) do not exist on `InfiniDopamineDecoderLayer`. The verified list for Qwen3.5-0.8B is: `lm_head`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `in_proj_w`, `in_proj_gate`, `out_proj`, `reward_gate_proj`, `reward_branch.output_proj`, `reward_branch.delta_layer.q_proj/k_proj/v_proj/w_proj/e_proj`, `reward_branch.delta_layer.baseline_tracker.alpha_proj`, `reward_branch.delta_layer.advantage_gate.advantage_proj`.

(20) **Direct-trained vs LoRA weights in PEFT.** Non-Linear modules in the parallel reward branch (RMSNorm, Parameter scalers, Conv1d) cannot receive LoRA adapters. They must be unfrozen directly. Because PEFT wraps `named_parameters()` with `base_model.model.` prefix and freezes non-target modules, the correct order is: (1) `ensure_all_trainable(model, missing)` BEFORE `get_peft_model`, then (2) `re_unfreeze_reward_branch(model)` AFTER `get_peft_model`.

(21) **`merge_and_unload()` preserves direct-trained weights.** LoRA adapters are merged into the base Linear weights; non-Linear params that were unfrozen directly remain in the merged model. Verified: merged model has `reward_branch`, total parameter count unchanged.

(22) **`SFTTrainer` kwarg rename.** `tokenizer=...` is deprecated in newer `transformers`; use `processing_class=tokenizer` to avoid `TypeError`.

(23) **Conditional optimizer fallback.** `optim="paged_adamw_8bit"` crashes if `bitsandbytes` is missing. Detect with `importlib.util.find_spec("bitsandbytes")` and fall back to `"adamw_torch"`.

(24) **Empty-row filtering after tokenization.** `tokenize_fn` must return empty lists for blank inputs, not None (IterableDataset.map cannot handle None). Then `.filter(lambda ex: len(ex.get("input_ids") or []) > 0)` drops them.

(25) **Canonical config construction.** `InfiniDopamineConfig(**hf_cfg.to_dict())` preserves all upstream fields verbatim. Hand-rolled `_as_obj` + `getattr` defaults silently drops fields and can pass wrong values (e.g. `hidden_size=1280` for vision when actual is 768). Patch only the parallel_reward keys into `text_config` dict after `to_dict()`.

(26) **HFIntegration registration guard.** `HFIntegration.register_infinidopamine_hf()` must run before `AutoConfig.from_pretrained` in any cell that might be rerun after a kernel restart. Make each config-using cell self-contained.

## Code-remediation TDD lessons (2026-09-07)

(27) **Import-time side effects need opt-in, not be silent.** Gated monkeypatches behind env vars (`QWENDOPAMINE_CPU_UNWRAP=1`); helper is a no-op without it.

(28) **Silently-true sentinels are bugs.** `AttributeError()` instance as class attr was truthy → silently picked MoE path. Replace with descriptor returning 0 so `> 0` checks evaluate to "not MoE".

(29) **Thread cache through every decode step.** `build_kv_caches` returning None re-initialised state on every step. Build per-layer dict, pass through `block.forward` → `self.attn(past_key_values=...)`.

(30) **Test brittle mocks at the gate, not the side effect.** M5 mock was intercepting real transformers import; refactor to assert the gate function.

(31) **Pre-existing tests pinning buggy behaviour must be re-pinned.** Update assertions; never silently keep them.

(32) **Pyrefly: explicit empty container types, never bare `[]` or `{}`.** Use `list[str]()`, `dict[str, torch.Tensor]()`.

(33) **Sequential phase-by-phase TDD; one commit per phase.** Multi-agent parallel caused file collisions.

(34) **Final state: 523 passed, 2 skipped, 6 deselected; ruff clean; pyrefly 0 errors.** Started at 491 baseline; added 32 new tests across 7 phases.
