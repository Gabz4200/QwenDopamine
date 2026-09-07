"""N5: reward.py docstring must not be self-contradictory.

The previous text said ``w_term[d] = omega_w_eff * write[d]`` while
also defining ``omega_w_eff = omega_w * write`` — those two lines
together imply the recurrence multiplies by ``write`` twice. The fix
collapses both into a single, unambiguous ``omega_w_eff = omega_w * write``
and uses that single name everywhere in the recurrence.
"""

from __future__ import annotations

import re


def test_reward_docstring_uses_single_canonical_name() -> None:
    """The module docstring must not contain ``w_term`` (the old alias)."""
    import qwendopamine.ops.reward as m

    doc = m.__doc__ or ""
    assert "w_term" not in doc, (
        "reward.py docstring still contains the old 'w_term' alias "
        "(self-contradictory with 'omega_w_eff = omega_w * write'). "
        "Review N5: use a single canonical name."
    )
    # The canonical definition must be present.
    assert "omega_w_eff" in doc, "omega_w_eff missing from reward.py docstring"
    # And the definition must match the math (omega_w * write).
    pattern = re.compile(
        r"omega_w_eff\[?[a-z]*\]?\s*=\s*omega_w\[?[a-z]*\]?\s*\*\s*write"
    )
    assert pattern.search(doc), (
        f"reward.py docstring does not contain 'omega_w_eff = omega_w * write'. "
        f"Got: {doc!r}"
    )
