"""Deprecated alias for :mod:`hand_derived_reference`.

This module used to be the canonical (hand-derived) third reference
for the Reinforced Delta memory core. The module was renamed to
:mod:`qwendopamine.models.reinforced.hand_derived_reference` to make
its role explicit (the file is a hand-written oracle, not a
generated reference).

This shim keeps the old import path working. New code should
"""

from __future__ import annotations

from qwendopamine.models.reinforced.hand_derived_reference import (
    canonical_delta_step,
    canonical_delta_step_with_grad,
)

__all__ = [
    "canonical_delta_step",
    "canonical_delta_step_with_grad",
]
