"""Utility functions for InfiniDopamine GatedDeltaNet.

Extracted from ``_gated_delta_net.py`` for size and SRP.
"""

from __future__ import annotations

import torch


def get_swa_mask(
    seq_len: int,
    sliding_window: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""Return a sliding-window causal mask, caching by (seq_len, sliding_window, device, dtype).

    Args:
        seq_len: Sequence length.
        sliding_window: Size of the sliding window.
        device: Optional device for the mask tensor.
        dtype: Optional dtype for the mask tensor.

    Returns:
        A causal mask tensor of shape ``(seq_len, seq_len)``.
    """
    # Generate lower-triangular mask with sliding window
    positions = torch.arange(seq_len, device=device)
    # Causal: position i can only attend to positions <= i
    # Sliding window: position i can only attend to positions >= i - sliding_window + 1
    range_matrix = positions[:, None] - positions[None, :]  # i - j
    mask = (range_matrix >= 0) & (range_matrix < sliding_window)
    return mask.float()