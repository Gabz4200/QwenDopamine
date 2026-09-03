"""Fake (meta) implementations for the public ops.

These are the shape and dtype stubs used by ``torch.compile``,
``FakeTensor``, and ``torch.library.opcheck`` to reason about the op
without invoking the Taichi kernel. They mirror the real op's
contract exactly.

The custom_op decorator on each op in
:mod:`qwendopamine.integrations.pytorch.custom_ops` already registers
the fake implementation; this module re-exports them so callers that
want the fake in isolation (e.g. for ``torch.library.opcheck``) can
import them directly.
"""

from __future__ import annotations

from qwendopamine.integrations.pytorch.custom_ops import (
    _chunk_gdn2_fake,
    _delta_core_step_fake,
    _recurrent_gdn2_fake,
)

__all__ = [
    "_chunk_gdn2_fake",
    "_delta_core_step_fake",
    "_recurrent_gdn2_fake",
]
