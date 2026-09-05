"""High-level API for the Taichi-accelerated GDN-2 kernels.

The functions in this module are drop-in replacements for the previous
``torch_chunk_gdn2`` / ``torch_recurrent_gdn2`` / ``triton.chunk_gdn2`` /
``triton.fused_recurrent_gdn2`` entry points. They use the same calling
convention (PyTorch tensors in BTHD layout) so the rest of the
``GatedDeltaNet2`` block does not need to know whether the underlying
engine is Taichi, Triton, or pure PyTorch.

Both the recurrent and the chunkwise paths are wrapped in
``torch.autograd.Function`` so training (backward) flows end-to-end.
The recurrent path stores every per-token state and per-token
activation and replays the token loop in reverse during ``backward``;
its per-step adjoint is implemented by
:func:`_kernels.launch_recurrent_step_bwd`. The chunkwise path keeps
the Taichi forward (production engine) and re-runs the equivalent
torch reference in reverse to obtain per-input gradients; that torch
reference matches the Taichi numerics because the public functions
operate on identical inputs and identical mathematical algorithm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_qk(
    q: torch.Tensor, k: torch.Tensor, apply: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optional L2-normalisation on Q and K. Matched to the legacy flags."""
    if not apply:
        return q, k
    q = F.normalize(q, p=2, dim=-1, eps=1e-6)
    k = F.normalize(k, p=2, dim=-1, eps=1e-6)
    return q, k


from qwendopamine.kernels.taichi._chunk_path import (
    _chunk_taichi_gdn2_inner,
    _ChunkTaichiGdn2Function,
    chunk_taichi_gdn2,
)
from qwendopamine.kernels.taichi._recurrent_path import (
    _recurrent_taichi_gdn2_inner,
    _RecurrentTaichiGdn2Function,
    recurrent_taichi_gdn2,
)

__all__ = [
    "_ChunkTaichiGdn2Function",
    "_RecurrentTaichiGdn2Function",
    "_chunk_taichi_gdn2_inner",
    "_normalize_qk",
    "_recurrent_taichi_gdn2_inner",
    "chunk_taichi_gdn2",
    "recurrent_taichi_gdn2",
]
