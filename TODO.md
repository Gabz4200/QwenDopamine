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

## Phase B — M5 + M6 + M7 + N1  (DONE)

- [x] M5: `QWENDOPAMINE_CPU_UNWRAP=1` opt-in env var; helper no-op without it.
- [x] M6: `load_gguf_weights` returns unexpected keys + logs warning.
- [x] M7 + N1: `_warn_mtp_drops` + per-partition INFO logs; `load_info` removed.

## Phase C — M9 + N5 + N6  (DONE)

- [x] M9: bare `except ...: pass` → `logger.debug(..., exc_info=True)`.
- [x] N5: `reward.py` docstring uses single canonical name `omega_w_eff`.
- [x] N6: `AttributeError()` sentinel replaced with `_UnsupportedAttr` descriptor
  returning 0 (parent config does `> 0` checks).

## Phase D — M11 + M12 + M14  (DONE)

- [x] M11: `build_kv_caches` returns per-layer dict for GDN-2 layers; block
  threads it as `past_key_values` so recurrent state persists across steps.
- [x] M12: KV cache trimmed to `max_seq_length` in `attention.py`.
- [x] M14: `norm_2` always allocated when MLP is present.

## Phase E — M13 + N8  (DONE)

- [x] M13: `_REGISTERED_AUTOGRAD` flipped to True only after successful
  `register_all_autograd()`; import-time call preserved for downstream
  test contract.
- [x] N8: dead `if TYPE_CHECKING: pass / else: pass` stub removed.

## Phase F — M15 + N9  (DONE)

- [x] M15: `trl.py` and `saving.py` deleted.
- [x] N9: `_broadcast_cond` consolidated to `components.broadcast_cond`
  (4 instances → 1).

## Phase G — N2, N3, N4, N11, N12  (DONE)

- [x] N2: perplexity raises `ValueError` on `total_tokens == 0`.
- [x] N3: `load_state_dict` warmup re-apply guarded by `step_count > 0`.
- [x] N4: `move_to_device` handles `NamedTuple` (via `_make`) and
  `frozenset`.
- [x] N11: `pyproject.toml` parsed via `tomllib` instead of substring
  assertion.
- [x] N12: `transformer` property returns cached `ModuleDict`.

## Phase H — Known / leave-as-is

- [x] N10: `RMSNormGatedNoCast` skips fp32 upcast — intentional. No action.

## Final

- [x] Full gate run: ruff, pyrefly, pytest `-m "not slow"`.
  - **pytest: 523 passed, 2 skipped, 6 deselected** (was 491 baseline; +32
    new tests).
  - ruff: **All checks passed**.
  - pyrefly: **0 errors**.
- [x] Update `MEMORY.md` / `memory_summary.md` with new lessons.
- [x] Hand off.

## Commit log

- `d0f0755` style: ruff + pyrefly clean; restore unexpected keys in M6 stub
- `177b73a` fix(g): perplexity zero-token, warmup guard, NamedTuple/frozenset, tomllib, transformer cache (N2, N3, N4, N11, N12)
- `f9f413d` fix(e): is_autograd_registered tracks actual state; remove TYPE_CHECKING stub (M13, N8)
- `9a10d92` fix(d): GDN2GPT — persist GDN-2 state across decode; trim KV cache; align norm_2 condition (M11, M12, M14)
- `31e3213` fix(m12): trim GDN2GPT KV cache to max_seq_length
- `5817bda` fix(m14): always allocate norm_2 in GDN2GPT block when MLP is present
- `58c95a4` fix(m9): debug-log parallel reward except path
- `da705a9` fix(weights): report unexpected GGUF keys; log mtp backfill; drop load_info (M6, M7, N1)
- `32ec52d` test(m5): replace brittle import-time mock with gate-level assertion
- `01673b4` feat(m5): gate qwen3_next CPU unwrap behind QWENDOPAMINE_CPU_UNWRAP opt-in env var
- `4f2e5e7` docs(todo): comprehensive handoff inventory
- `e9fcec5` fix(gdn2): propagate chunk_size to torch fallback; add initial_state to accel arg indices (M4, M10)
- Earlier handoff commits 30f6e8d, 234ac64, 5335cbb, 282d3bc, 92cbcae
