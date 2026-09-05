"""GDN-2 recurrent forward operations.

Implements the ``qwendopamine::recurrent_gdn2`` and
``qwendopamine::recurrent_gdn2_with_state`` custom ops split into
separate body, fake, and public API modules.

Schema contract
---------------
Each op follows the "stable schema" rule from the
`Custom Python Operators tutorial
<https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html>`_:

    - the number of return Tensors is fixed per op (no maybe-out);
    - the returned Tensors do not alias any input Tensor;
    - the fake kernel matches the real kernel's output metadata
      (shape, dtype, device, layout, strides, storage offset);
    - the fake kernel may inspect metadata but must not read data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qwendopamine.integrations.pytorch.register import _route_to_active_device

if TYPE_CHECKING:
    pass
else:
    from torch.library import custom_op  # runtime import for decorators


def _recurrent_gdn2_body(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    from qwendopamine.ops import recurrent_taichi_gdn2

    out, _state = recurrent_taichi_gdn2(
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


def _recurrent_gdn2_with_state_body(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> list[torch.Tensor]:
    from qwendopamine.ops import recurrent_taichi_gdn2

    out, state = recurrent_taichi_gdn2(
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
    "qwendopamine::recurrent_gdn2",
    mutates_args=(),
    device_types="cpu",
)
def recurrent_gdn2_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """GDN-2 single-token recurrent forward; returns the output only."""
    result = _route_to_active_device(
        _recurrent_gdn2_body, q, k, v, g, b, w, initial_state
    )
    assert isinstance(result, torch.Tensor)
    return result


@recurrent_gdn2_op.register_fake
def _recurrent_gdn2_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fake implementation. Output is a fresh contiguous ``[B, T, H, V]``."""
    return torch.empty(
        q.shape[0],
        q.shape[1],
        q.shape[2],
        v.shape[-1],
        dtype=q.dtype,
        device=q.device,
    )


@custom_op(
    "qwendopamine::recurrent_gdn2_with_state",
    mutates_args=(),
    device_types="cpu",
)
def recurrent_gdn2_with_state_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """GDN-2 single-token recurrent forward; returns ``[output, final_state]``."""
    result = _route_to_active_device(
        _recurrent_gdn2_with_state_body, q, k, v, g, b, w, initial_state
    )
    assert isinstance(result, list)
    return result


@recurrent_gdn2_with_state_op.register_fake
def _recurrent_gdn2_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Fake implementation. Returns ``[out, state]`` with correct strides."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    out = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    state = torch.empty(B, H, K, V, dtype=q.dtype, device=q.device)
    return [out, state]
