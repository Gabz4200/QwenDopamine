"""InfiniDopamineRMSNorm: RMS normalization wrapper.

Moved from ``decoder_layer.py`` for size.
"""

from __future__ import annotations

from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm


class InfiniDopamineRMSNorm(Qwen3NextRMSNorm):
    r"""InfiniDopamineRMSNorm: RMS normalization (adapter for :class:`Qwen3NextRMSNorm`).

    This is a framework adapter. It exists so the rest of the codebase
    can depend on a project-owned abstraction instead of the concrete
    ``transformers`` implementation. If the upstream RMSNorm API changes,
    only this module needs updating.
    """
