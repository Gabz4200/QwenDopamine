
## Notebook/Kaggle and PEFT pipeline lessons (2026-09-05)

- Kaggle notebook install cells must avoid `git+https` pip URLs; use GitHub Releases wheel + `importlib.util.find_spec` guard.
- Pin deps to actually-existing versions: `transformers>=4.40.0,<5.0.0`, `torch>=2.0.0`, `unsloth>=4.7` per pyproject extras.
- Root `OUTPUT_DIR` at `/kaggle/working` and append `os.getpid()` to avoid collisions on reruns.
- `InfiniDopamineConfig(**hf_cfg.to_dict())` is the canonical config construction; do not hand-roll defaults via `_as_obj`.
- Verified PEFT pipeline: 27 non-Linear missing keys are direct-trained; 16 LoRA targets resolve to `nn.Linear`; re-unfreeze reward_branch after PEFT wrap; `merge_and_unload()` preserves direct-trained weights and total param count.
- `SFTTrainer` now uses `processing_class=`; `optim="paged_adamw_8bit"` needs `bitsandbytes` fallback.
- Tokenize empty rows as empty lists, then filter; never return None from `IterableDataset.map`.
- Notebook passes all quality gates: ruff, pyrefly 0 errors, jupytext sync, syntax valid.

## Code-remediation TDD lessons (2026-09-07)

- **Import-time side effects need opt-in, not be silent.** The qwen3_next CPU-unwrap monkeypatch (M5) fired at every `import qwendopamine.models.*` and silently rewrote upstream `transformers` functions. Gated behind `QWENDOPAMINE_CPU_UNWRAP=1` env var; helper is a no-op without it.

- **Silently-true sentinels are bugs.** The `AttributeError()` instance trick for MoE config fields (N6) was truthy → InfiniDopamine silently picked the MoE path. Replace with a `_UnsupportedAttr` descriptor that returns 0 (or another sane default) so parent config `> 0` checks evaluate to "not MoE" without raising.

- **Thread cache through every decode step.** `build_kv_caches` returning `None` for GDN-2 layers (M11) meant every decode step re-initialised the recurrent state. Build a per-layer `dict` and pass it through `block.forward` → `self.attn(past_key_values=...)`.

- **Test brittle mocks at the gate, not the side effect.** The M5 test originally tried to intercept the real transformers import; that path is fragile. Refactor to assert the gate function (`_should_unwrap_for_cpu`) returns the right value — the seam that the fix actually touches.

- **Pre-existing tests pinning buggy behaviour must be re-pinned, not deleted.** `test_when_compute_perplexity_empty_dataloader_then_returns_finite_value` (N2) and the MoE weight-loading tests pinned the old broken contracts. Update the assertions to the new contract; never silently keep them.

- **Pyrefly: explicit empty container types, never bare `[]` or `{}`.** Use `list[str]()`, `dict[str, torch.Tensor]()`, etc. The `# type: ignore[arg-type,return-value]` for monkeypatch lambdas is fine.

- **RED-first, one phase at a time.** The original multi-agent parallel approach caused file collisions in the same working tree. Sequential, one phase at a time, with each phase TDD'd and committed before moving on, kept the work clean.

- **Test order can flip flaky AOT tests.** `test_opcheck_battery[chunk_gdn2-False]` passes in a full suite run but fails in isolation. Document and accept; it's a test-order dependency on taichi init, not a regression.

- **Final state: 523 passed, 2 skipped, 6 deselected; ruff clean; pyrefly 0 errors.** Started at 491 passed baseline; added 32 new tests across 7 phases (M4–M15, N1–N12, plus the handoff's H1–H5, M8, N7).
