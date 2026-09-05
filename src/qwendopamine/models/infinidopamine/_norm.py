"""InfiniDopamineRMSNorm: RMS normalization wrapper.

Moved from ``decoder_layer.py`` for size.
"""

from __future__ import annotations

from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm


class InfiniDopamineRMSNorm(Qwen3NextRMSNorm):
    r"""InfiniDopamineRMSNorm: RMS normalization (inherits from
    :class:`Qwen3NextRMSNorm`).
    """
