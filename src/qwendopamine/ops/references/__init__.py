"""Readable PyTorch reference implementations.

These live beside the public ops in :mod:`qwendopamine.ops` (not in the
``models/`` tree) so any caller can read the math without pulling in
the full InfiniDopamine model stack. Production paths (the Taichi
kernels and the ``torch_chunk_gdn2`` / ``torch_recurrent_gdn2`` ops)
must match these references within numerical tolerance.

Use these for:

    numerical correctness checks;
    randomised property tests;
    ``torch.autograd.gradcheck``;
    CPU vs GPU comparison;
    edge cases;
    validating new Taichi implementations.
"""

from __future__ import annotations

from qwendopamine.ops.references.gdn2_oracle import gdn2_oracle_step
from qwendopamine.ops.references.gdn2_reference import (
    gdn2_reference_sequence,
    gdn2_reference_step,
    gdn2_reference_step_with_grad,
)
from qwendopamine.ops.references.reward_reference import (
    reward_reference_step,
    reward_reference_step_with_grad,
)

__all__ = [
    "gdn2_oracle_step",
    "gdn2_reference_sequence",
    "gdn2_reference_step",
    "gdn2_reference_step_with_grad",
    "reward_reference_step",
    "reward_reference_step_with_grad",
]
