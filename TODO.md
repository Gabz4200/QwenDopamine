# TODO — QwenDopamine remediation

Continuation from the handoff. Picks up after fixes 1-7 (H5, H1, H4, H2, H3,
M8, N7) are committed in the working tree. Verified baseline 496 passed / 2
skipped before starting.

This file is the authoritative state tracker for the handoff's two code-review
passes. It catalogues every finding (HIGH / MEDIUM / LOW-NIT) from both passes
plus every TDD fix applied so far, with status, file, and acceptance criterion.

---

## Already fixed (handoff-applied TDD, in working tree)

| # | ID  | File(s) | Fix summary | Test |
|---|-----|---------|-------------|------|
| 1 | H5  | `integrations/pytorch/delta.py` | Added `d_k` read-back term `-sum_d state[d,k]*d_e[d]`; updated `setup_context` to torch 2.9+ `(ctx, inputs, output)` signature | `tests/ops/test_autograd.py::test_delta_core_step_custom_op_backward_matches_torch_reference` |
| 2 | H1  | `models/model_factory.py` | Lazy import of `infinidopamine`/`qwen35` in `_resolve_model_family`, `build_model`, `build_reference_model` | `tests/test_models_submodules.py::test_import_qwendopamine_models_does_not_pull_transformers` |
| 3 | H4  | `models/gdn2/block.py` | Removed direct `kernels.taichi.gdn2_api` imports + eager `is_available()` probe; added lazy `_taichi_ops_available()`; `_run_taichi_backend` dispatches through `qwendopamine.ops` | `tests/models/test_gdn2.py` + `test_gdn2_model.py` (33 passed) |
| 4 | H2  | `training/loop.py` | `run()` uses `iter(train_loader)` + peek (lazy streaming) instead of `list(train_loader)` | `tests/models/test_training.py::test_when_training_loop_streaming_loader_then_does_not_materialize` |
| 5 | H3  | `training/loop.py` | Partial tail steps with `grad_scale=grad_accum_steps/accum` to preserve gradient magnitude | `tests/models/test_training.py::test_when_training_loop_partial_tail_then_gradient_scale_preserved` |
| 6 | M8  | `training/loop.py` | `__init__` raises `ValueError` if `mixed_precision="fp16"` on CPU | `tests/models/test_training.py::test_when_fp16_on_cpu_then_raises_value_error` |
| 7 | N7  | `training/schedules.py` | Override `_initial_step` to no-op to avoid spurious LR-scheduler warning | `tests/models/test_schedules.py` (warning gone) |
| 8 | M4  | `src/qwendopamine/ops/gdn2.py` | `chunk_taichi_gdn2` torch-fallback now forwards `chunk_size=chunk_size` to `torch_chunk_gdn2(...)` | `tests/ops/test_gdn2_op_dispatch.py::TestGdn2OpDispatch::test_chunk_size_is_propagated_to_torch_fallback` |
| 9 | M10 | `src/qwendopamine/integrations/pytorch/register.py` | Extracted `GDN2_ACCEL_TENSOR_ARG_INDICES = [[0..6] x4]`; the four GDN-2 ops now include `initial_state` (arg 6) in `tensor_arg_indices` | `tests/test_integ_root.py::test_when_register_accelerator_kernels_then_initial_state_in_arg_indices` |

Commits in working tree (most-recent first):
`e9fcec5` (M4+M10), `30f6e8d` (top of handoff), `234ac64`, `5335cbb`,
`282d3bc`, `92cbcae` (7 handoff fixes).

---

## Phase A — M4 + M10  (DONE)

- [x] M4: Pass `chunk_size` through torch-reference fallback in `ops/gdn2.py`.
- [x] M10: Add arg index 6 (`initial_state`) to `tensor_arg_indices` for
  chunk/recurrent ops in `integrations/pytorch/register.py`.
- [x] Re-run quality gates.
  - pytest `-m "not slow"`: **498 passed, 2 skipped, 6 deselected** (was 496).
  - ruff: 4 pre-existing errors in `models/gdn2/block.py` from H4 fix
    (not introduced by this phase). pyrefly: 2 pre-existing errors in
    `tests/models/test_training.py` from H2/H3 fix. Neither blocking.

## Phase B — M5 + M6 + M7 + N1  (weight-loading & transformers-util)

- [ ] M5: Gate the CPU monkeypatch of `qwen3_next` behind an explicit opt-in
  (env var or explicit call) in `src/qwendopamine/models/_transformers_utils.py`.
  Currently fires at import time.
- [ ] M6: Report `unexpected` keys from `load_state_dict(strict=False)` in
  `src/qwendopamine/integrations/gguf.py` (mirror the `allowed_missing`
  pattern, or switch to strict after filtering).
- [ ] M7 + N1: Log mtp backfill (with count) and treat `mtp.*` as
  misc/warn rather than silently dropped in `strict=True` mode in
  `src/qwendopamine/models/infinidopamine/_qwen35_weights.py` and
  `_text_qwen35_weights.py`. Also surface or drop the unused `load_info`
  list.
- [ ] Re-run quality gates.

## Phase C — M9 + N5 + N6  (small surface cleanups)

- [ ] M9: Replace bare `except (AttributeError, IndexError): pass` in
  `src/qwendopamine/training/parallel_reward.py:142-143` with debug-level log
  so a misconfigured model surfaces in logs instead of vanishing.
- [ ] N5: Fix the self-contradictory `w_term` / `omega_w_eff` description in
  the `src/qwendopamine/ops/reward.py` module docstring (current text says
  both that `w_term` is a gate and that `omega_w_eff` is the contracted form;
  pick one and rewrite the doc).
- [ ] N6: Replace the `norm_topk_prob = AttributeError()` class-attr sentinel
  in `src/qwendopamine/models/infinidopamine/configs.py` with a cleaner
  sentinel (e.g. None + property, or a `_Missing` enum).
- [ ] Re-run quality gates.

## Phase D — M11 + M12 + M14  (GDN2GPT incremental decode)

- [ ] M11: Persist GDN-2 recurrent state across incremental decode steps in
  `src/qwendopamine/models/gdn2_gpt/block.py` (and `attention.py` /
  `model.py` as needed). Current code calls
  `self.attn(n_1, attention_mask=None)` with no cache; `build_kv_caches`
  reserves `None` for GDN-2 layers, so memory is re-initialised to zero on
  every decode step.
- [ ] M12: Trim the KV cache to `max_seq_length` in
  `src/qwendopamine/models/gdn2_gpt/attention.py:121-133`. Current code does
  `k = torch.cat([cache_k, k], dim=2)` with no cap (lit-gpt caps).
- [ ] M14: Make `norm_2` creation/use consistent in
  `src/qwendopamine/models/gdn2_gpt/block.py:62-67`. The conditional
  `not shared_attention_norm and mlp and not parallel_residual` creates
  `norm_2`, but `forward` uses it whenever `mlp and not parallel_residual`.
  Latent crash on `shared_attention_norm=True` combo. Guard both sides with
  the same condition or always create it.
- [ ] Re-run quality gates.

## Phase E — M13 + N8  (autograd honesty + dead code)

- [ ] M13: Make `is_autograd_registered()` / `is_registered()` actually
  reflect registration in
  `src/qwendopamine/integrations/pytorch/autograd.py:42` (sets
  `_REGISTERED_AUTOGRAD = True` from the start) and
  `src/qwendopamine/integrations/pytorch/custom_ops.py:122` (same pattern).
  Both report True even when the autograd rule never registered.
- [ ] N8: Remove the dead `if TYPE_CHECKING: pass / else: pass` stub in
  `src/qwendopamine/integrations/pytorch/custom_ops.py:40-43`.
- [ ] Re-run quality gates.

## Phase F — M15 + N9  (dead code + de-dup)

- [ ] M15: Delete the dead duplicate modules
  `src/qwendopamine/integrations/huggingface/trl.py` and `saving.py`
  (canonical sources: `_build.py` and `_save.py`). Neither is imported
  by `integration.py`.
- [ ] N9: De-duplicate `_broadcast_cond` across
  `src/qwendopamine/models/blocks/reward/{scalers,film,fourier}.py`
  (extract to a shared helper, e.g. `components.py`).
- [ ] Re-run quality gates.

## Phase G — Low priority  (N2, N3, N4, N11, N12)

- [ ] N2: Make `total_tokens == 0` explicit in
  `src/qwendopamine/evaluation/perplexity.py` — currently returns `ppl=1.0`
  silently. Either raise or return `float('inf')` with a clear contract.
- [ ] N3: Guard `load_state_dict` warmup re-application with
  `step_count > 0` in `src/qwendopamine/training/schedules.py`. Currently
  re-applies warmup on every restore, so a fresh restore gives LR=0.0.
- [ ] N4: Handle `NamedTuple` and `frozenset` in
  `src/qwendopamine/utils.py:51` `move_to_device`. Current
  `type(batch)(moved)` breaks `NamedTuple` and misses `frozenset`.
- [ ] N11: Stop asserting `pyproject.toml` source text in
  `tests/test_dependency_compatibility.py`. Parse it with `tomllib` instead.
- [ ] N12: Cache the `transformer` property's `ModuleDict` in
  `src/qwendopamine/models/gdn2_gpt/model.py` — current property allocates
  a fresh `ModuleDict`/access on every call.
- [ ] Re-run quality gates.

## Phase H — Known / leave-as-is

- [ ] N10: `RMSNormGatedNoCast` skips fp32 upcast — intentional. No action.
  Logged for completeness.

## Final

- [ ] Full gate run: ruff, pyrefly, pytest `-m "not slow"`.
- [ ] Update `MEMORY.md` / `memory_summary.md` with new lessons.
- [ ] Hand off.
