"""Output projection helpers for GDN-2.

Extracted from :mod:`block` for modularity. These are stateless
functions parameterised by the output projection layers so the wrapping
:class:`~qwendopamine.models.gdn2.block.GatedDeltaNet2` can delegate to them.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from qwendopamine.models.gdn2.recurrence.packing import pad_input


def compute_output(
    hidden_states: torch.Tensor,
    o: torch.Tensor,
    g_proj: nn.Sequential,
    o_norm: nn.Module,
    o_proj: nn.Linear,
    head_v_dim: int,
    is_padded: bool,
    indices: torch.Tensor | None,
    batch: int,
    seq_len: int,
) -> torch.Tensor:
    r"""compute_output(hidden_states, o, g_proj, o_norm, o_proj, head_v_dim, is_padded, indices, batch, seq_len) -> torch.Tensor

    Apply the output gate, normalization, projection, and optional unpadding.

    Args:
        hidden_states (torch.Tensor): Input ``[B, T, D]``.
        o (torch.Tensor): Mixer output ``[B, T, H, V]``.
        g_proj (nn.Sequential): Output gate projection.
        o_norm (nn.Module): Output normalization with gating.
        o_proj (nn.Linear): Output projection.
        head_v_dim (int): Per-head value dimension.
        is_padded (bool): Whether the input was padded.
        indices (torch.Tensor | None): Unpad indices when ``is_padded``.
        batch (int): Batch size.
        seq_len (int): Padded sequence length.

    Returns:
        torch.Tensor: Output ``[B, T, D]``.
    """
    gate = rearrange(g_proj(hidden_states), "... (h d) -> ... h d", d=head_v_dim)
    if o is None:
        raise RuntimeError("GDN-2 backend returned None output.")
    o = o_norm(o, gate)
    o = rearrange(o, "... h d -> ... (h d)")
    out = o_proj(o)
    if is_padded and indices is not None:
        out = pad_input(out.squeeze(0), indices, batch, seq_len)
    result: torch.Tensor = out
    return result


__all__ = ["compute_output"]
