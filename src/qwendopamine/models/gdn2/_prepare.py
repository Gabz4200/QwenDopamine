"""Token preparation and backend-dispatch helpers for GDN-2.

Extracted from :mod:`block` for modularity. These helpers receive the
:class:`~qwendopamine.models.gdn2.block.GatedDeltaNet2` instance as
``layer`` so they can read configuration attributes (``head_v_dim``,
``num_heads``, ``num_v_heads``, ``allow_neg_eigval``) without going
through ``self``.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from einops import rearrange, repeat


def prepare_tokens(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    head_v_dim: int,
    head_k_dim: int,
    num_heads: int,
    num_v_heads: int,
    allow_neg_eigval: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    r"""prepare_tokens(q, k, v, g, b, w, head_v_dim, head_k_dim, num_heads, num_v_heads, allow_neg_eigval) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    Rearrange projections into per-head layout and apply value-head grouping.

    Args:
        q (torch.Tensor): Query ``[B, T, H, K]``.
        k (torch.Tensor): Key ``[B, T, H, K]``.
        v (torch.Tensor): Value ``[B, T, H, V]``.
        g (torch.Tensor): Decay ``[B, T, H, K]``.
        b (torch.Tensor): Erase gate ``[B, T, H, K]``.
        w (torch.Tensor): Write gate ``[B, T, H, V]``.
        head_v_dim (int): Per-head value dimension.
        head_k_dim (int): Per-head key dimension.
        num_heads (int): Number of query heads.
        num_v_heads (int): Number of value heads.
        allow_neg_eigval (bool): Whether to allow negative eigenvalues.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ``(q, k, v, g, b, w)`` in per-head layout.
    """
    q = rearrange(q, "... (h d) -> ... h d", d=head_k_dim)
    k = rearrange(k, "... (h d) -> ... h d", d=head_k_dim)
    g = rearrange(g, "... (h d) -> ... h d", d=head_k_dim)
    v = rearrange(v, "... (h d) -> ... h d", d=head_v_dim)
    b = rearrange(b, "... (h d) -> ... h d", d=head_k_dim)
    w = rearrange(w, "... (h d) -> ... h d", d=head_v_dim)

    if num_v_heads > num_heads:
        groups = num_v_heads // num_heads
        q, k, g, b = (
            repeat(x, "... h d -> ... (h g) d", g=groups) for x in (q, k, g, b)
        )

    if allow_neg_eigval:
        b = b * 2.0
    return q, k, v, g, b, w


def dispatch_backend(
    layer: Any,
    backend: str,
    mode: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    recurrent_state: torch.Tensor | None,
    use_cache: bool,
    cu_seqlens: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""dispatch_backend(layer, backend, mode, q, k, v, g, b, w, recurrent_state, use_cache, cu_seqlens) -> tuple[torch.Tensor, torch.Tensor | None]

    Dispatch the forward pass to the selected GDN-2 backend.

    Args:
        layer (Any): The :class:`GatedDeltaNet2` instance (passed for backend method access).
        backend (str): Selected backend identifier.
        mode (str): Selected mode (``"chunk"`` or ``"fused_recurrent"``).
        q (torch.Tensor): Query ``[B, T, H, K]``.
        k (torch.Tensor): Key ``[B, T, H, K]``.
        v (torch.Tensor): Value ``[B, T, H, V]``.
        g (torch.Tensor): Decay ``[B, T, H, K]``.
        b (torch.Tensor): Erase gate ``[B, T, H, K]``.
        w (torch.Tensor): Write gate ``[B, T, H, V]``.
        recurrent_state (torch.Tensor | None): Initial state.
        use_cache (bool): Whether to return the final state.
        cu_seqlens (Any): Cumulative sequence lengths.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: ``(output, final_state)``.
    """
    if backend == "taichi":
        return cast(
            tuple[torch.Tensor, torch.Tensor | None],
            layer._run_taichi_backend(
                mode, q, k, v, g, b, w, recurrent_state, use_cache
            ),
        )
    return cast(
        tuple[torch.Tensor, torch.Tensor | None],
        layer._run_torch_backend(
            backend, mode, q, k, v, g, b, w, recurrent_state, use_cache
        ),
    )


__all__ = ["dispatch_backend", "prepare_tokens"]
