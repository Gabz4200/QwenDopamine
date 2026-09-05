"""GDN-2 chunkwise forward operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qwendopamine.integrations.pytorch.register import _route_to_active_device

if TYPE_CHECKING:
    pass
else:
    from torch.library import custom_op  # runtime import for decorators


def _chunk_gdn2_body(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    from qwendopamine.ops import chunk_taichi_gdn2

    out, _state = chunk_taichi_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=False,
    )
    return out.contiguous()


def _chunk_gdn2_with_state_body(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> list[torch.Tensor]:
    from qwendopamine.ops import chunk_taichi_gdn2

    out, state = chunk_taichi_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=True,
    )
    state_safe = state.contiguous() if state is not None else q.new_empty(0)
    return [out.contiguous(), state_safe]


@custom_op(
    "qwendopamine::chunk_gdn2",
    mutates_args=(),
    device_types="cpu",
)
def chunk_gdn2_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    result = _route_to_active_device(_chunk_gdn2_body, q, k, v, g, b, w, initial_state)
    assert isinstance(result, torch.Tensor)
    return result


@chunk_gdn2_op.register_fake
def _chunk_gdn2_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(
        q.shape[0],
        q.shape[1],
        q.shape[2],
        v.shape[-1],
        dtype=q.dtype,
        device=q.device,
    )


@custom_op(
    "qwendopamine::chunk_gdn2_with_state",
    mutates_args=(),
    device_types="cpu",
)
def chunk_gdn2_with_state_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    result = _route_to_active_device(
        _chunk_gdn2_with_state_body, q, k, v, g, b, w, initial_state
    )
    assert isinstance(result, list)
    return result


@chunk_gdn2_with_state_op.register_fake
def _chunk_gdn2_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    out = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    state = torch.empty(B, H, K, V, dtype=q.dtype, device=q.device)
    return [out, state]
