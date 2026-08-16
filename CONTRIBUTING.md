# Contributing

This project hosts multiple research architectures. This guide documents the
sub-project that repeatedly tends to accumulate complexity: **GDN-2** (Gated
DeltaNet 2) and its host models.

## The one rule that keeps GDN-2 maintainable

GDN-2 has three layers, and they must stay separate:

1. **Mathematics** — `src/qwendopamine/models/gdn2/gdn2.py`
   - `torch_recurrent_gdn2` (the token-by-token oracle),
   - `torch_chunk_gdn2` (the chunkwise/WY formulation, the training reference),
   - `compute_gdn2_*` helpers (WY solve, intra-chunk scores), the decay-gate
     activation, and the pure-PyTorch short-conv / gated-RMSNorm modules.
   - This layer must **never** depend on hardware-specific libraries.
2. **Model integration** — the host `SurpriseGPT` (`surprise_gpt/`) and the
   `Qwen3_5`/`gdn2` blocks.
   - These decide *which* layers are GDN-2 and own caches/short-conv state.
   - They must **not** contain GDN-2 mathematics.
3. **Execution backend** — the `backend=` selector on `GatedDeltaNet2`.
   - `auto | torch | torch-chunk | torch-recurrent | compiled | triton | fla`.
   - `resolve_gdn2_backend` is the **only** place device/training/sequence-length
     decisions live. The maths never branches on device.
   - Backends may optimize the maths but must preserve its public semantics.

## Homework

- When you optimize an implementation and it gets harder to read, quarantine
  that complexity behind a backend rather than spreading it through the core
  maths.
- Validate every accelerated backend against the pure-PyTorch oracle:

  ```
  uv run python -m pytest tests/models/test_gdn2.py tests/models/test_gdn2_params.py
  ```

  The tests pin: `torch_recurrent == torch_chunk` across chunk sizes and
  partial final chunks, gradient equality between the two paths, and exact
  parameter counts for the host model.
- Keep the decay activation computed in fp32 (`fp32_decay=True`) for anything
  numerically sensitive, per the GDN-2 paper (Sec. D).