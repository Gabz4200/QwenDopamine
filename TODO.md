# TODO — QwenDopamine remediation

Continuation from the handoff. Picks up after fixes 1-7 (H5, H1, H4, H2, H3,
M8, N7) are committed in the working tree. Verified baseline 496 passed / 2
skipped before starting.

## Phase A — M4 + M10
- [x] M4: Pass `chunk_size` through torch-reference fallback in `ops/gdn2.py`.
  - RED: `tests/ops/test_gdn2_op_dispatch.py::test_chunk_size_is_propagated_to_torch_fallback`
  - GREEN: added `chunk_size=chunk_size` to `torch_chunk_gdn2(...)` call.
- [x] M10: Add arg index 6 (`initial_state`) to `tensor_arg_indices` for
  chunk/recurrent ops in `integrations/pytorch/register.py`.
  - RED: `tests/test_integ_root.py::test_when_register_accelerator_kernels_then_initial_state_in_arg_indices`
  - GREEN: extracted `GDN2_ACCEL_TENSOR_ARG_INDICES` constant with `[0..6]`,
    wired into `per_op_specs` for the four GDN-2 ops.
- [x] Re-run quality gates.
  - pytest `-m "not slow"`: **498 passed, 2 skipped, 6 deselected** (was 496).
  - ruff: 4 pre-existing errors in `models/gdn2/block.py` from H4 fix
    (not introduced by this phase). pyrefly: 2 pre-existing errors in
    `tests/models/test_training.py` from H2/H3 fix. Neither blocking.

## Phase B — M5 + M6 + M7 + N1
- [ ] M5: Gate CPU monkeypatch of `qwen3_next` behind opt-in
  (`_transformers_utils.py`).
- [ ] M6: Report unexpected keys in `integrations/gguf.py` loader.
- [ ] M7 + N1: Log mtp backfill + drop unused `load_info` in
  `_qwen35_weights.py` / `_text_qwen35_weights.py`.
- [ ] Re-run quality gates.

## Phase C — M9 + N5 + N6
- [ ] M9: Log at debug level in `parallel_reward.py:142-143`.
- [ ] N5: Fix self-contradictory `w_term` / `omega_w_eff` docstring in
  `ops/reward.py`.
- [ ] N6: Replace `AttributeError()` class-attr sentinel in
  `infinidopamine/configs.py`.
- [ ] Re-run quality gates.

## Phase D — M11 + M12 + M14 (GDN2GPT decode)
- [ ] M11: Persist GDN-2 recurrent state across decode steps in
  `gdn2_gpt/block.py` (and `attention.py` / `model.py` as needed).
- [ ] M12: Trim KV cache to `max_seq_length` in `gdn2_gpt/attention.py`.
- [ ] M14: Make `norm_2` creation/use consistent in `gdn2_gpt/block.py`.
- [ ] Re-run quality gates.

## Phase E — M13 + N8 (autograd honesty)
- [ ] M13: Make `is_autograd_registered()` / `is_registered()` actually
  reflect registration in `integrations/pytorch/{autograd,custom_ops}.py`.
- [ ] N8: Remove dead `if TYPE_CHECKING: pass / else: pass` stub.
- [ ] Re-run quality gates.

## Phase F — M15 + N9 (dead code + de-dup)
- [ ] M15: Delete dead `integrations/huggingface/trl.py` and `saving.py`
  (canonical: `_build.py`, `_save.py`).
- [ ] N9: De-duplicate `_broadcast_cond` across `blocks/reward/{scalers,film,fourier}.py`.
- [ ] Re-run quality gates.

## Phase G — Low priority (N2, N3, N4, N11, N12)
- [ ] N2: Raise or `inf` with contract on `total_tokens == 0` in `perplexity.py`.
- [ ] N3: Guard `load_state_dict` warmup re-application with `step_count > 0`.
- [ ] N4: Handle `NamedTuple`/`frozenset` in `utils.py:move_to_device`.
- [ ] N11: Parse `pyproject.toml` instead of asserting source text.
- [ ] N12: Cache `transformer` property `ModuleDict` in `gdn2_gpt/model.py`.
- [ ] Re-run quality gates.

## Final
- [ ] Full gate run: ruff, pyrefly, pytest `-m "not slow"`.
- [ ] Update MEMORY.md / memory_summary.md with new lessons.
- [ ] Hand off.