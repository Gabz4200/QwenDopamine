"""GDN-2 recurrence kernels.

Pure PyTorch implementations of the Gated DeltaNet-2 matrix-state update:

- ``recurrent``  — token-by-token sequential engine (single-step + loop)
- ``chunk``      — chunkwise WY representation for training (paper Appendix A)
- ``packing``    — packed-sequence padding helpers (drop / gather / scatter)
"""

from qwendopamine.models.gdn2.recurrence.chunk import (
    compute_gdn2_intra_chunk_scores,
    compute_gdn2_wy_coefficients,
    torch_chunk_gdn2,
)
from qwendopamine.models.gdn2.recurrence.packing import (
    get_unpad_data,
    index_first_axis,
    pad_input,
)
from qwendopamine.models.gdn2.recurrence.recurrent import (
    gated_delta_2_step,
    process_sequence_recurrence,
    torch_recurrent_gdn2,
)

__all__ = [
    "compute_gdn2_intra_chunk_scores",
    "compute_gdn2_wy_coefficients",
    "gated_delta_2_step",
    "get_unpad_data",
    "index_first_axis",
    "pad_input",
    "process_sequence_recurrence",
    "torch_chunk_gdn2",
    "torch_recurrent_gdn2",
]
