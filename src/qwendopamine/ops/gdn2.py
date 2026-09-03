"""GDN-2 public ops with Taichi fallback.

Backend choice is delegated to Taichi: the kernel runtime picks CUDA →
Vulkan → Metal/OpenGL → CPU on its own. When Taichi is unavailable,
this module falls back to the pure-PyTorch reference.
"""

import torch

from qwendopamine.kernels.taichi import is_available as _is_available
from qwendopamine.models.gdn2.recurrence.chunk import torch_chunk_gdn2
from qwendopamine.models.gdn2.recurrence.recurrent import torch_recurrent_gdn2


def chunk_taichi_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    chunk_size: int = 64,
    **_: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""GDN-2 chunkwise forward + backward.

    Executes one chunkwise GDN-2 recurrence step. Delegates to the Taichi
    kernel when the runtime is available; otherwise falls back to the
    pure-PyTorch reference (:func:`torch_chunk_gdn2`). Output is always
    returned as a contiguous tensor so downstream ``torch.compile`` /
    ``opcheck`` callers see a stable memory layout.

    The chunkwise path reduces the ``O(T)`` token serialism to ``O(T/C)``
    chunk steps while remaining hardware-agnostic (paper Appendix A).

    Args:
        q: Queries ``[B, T, H, K]``.
        k: Keys ``[B, T, H, K]``.
        v: Values ``[B, T, H, V]``.
        g: Log-decay gate ``[B, T, H, K]``.
        b: Channel-wise erase gate ``[B, T, H, K]``.
        w: Channel-wise write gate ``[B, T, H, V]``.
        initial_state: Optional initial recurrent state ``[B, H, K, V]``.
        output_final_state: If ``True``, return the final state.
        use_qk_l2norm_in_kernel: Apply L2 normalization to q/k.
        chunk_size: Size of each chunk ``C`` for the WY recurrence.

    Returns:
        out: Output tensor ``[B, T, H, V]``.
        state: Final recurrent state ``[B, H, K, V]`` if
            ``output_final_state`` is ``True``, else ``None``.
    """
    if _is_available():
        from qwendopamine.kernels.taichi.gdn2_api import (
            chunk_taichi_gdn2 as _fn,
        )

        out, state = _fn(
            q,
            k,
            v,
            g,
            b,
            w,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            chunk_size=chunk_size,
        )
        # Defensive clone: the kernel may return a tensor that shares
        # storage with ``initial_state`` on some backends. The public op
        # must not return a value aliased to an input (the
        # ``custom_op`` schema contract requires this).
        if state is not None and state.data_ptr() == (
            initial_state.data_ptr() if initial_state is not None else 0
        ):
            state = state.clone()
        return (out.contiguous(), state) if out is not None else (q.new_empty(0), state)
    out, state = torch_chunk_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    if state is not None and state.data_ptr() == (
        initial_state.data_ptr() if initial_state is not None else 0
    ):
        state = state.clone()
    return (out.contiguous(), state) if out is not None else (q.new_empty(0), state)


def recurrent_taichi_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    **_: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""GDN-2 single-token recurrent forward + backward.

    Executes the GDN-2 recurrence token-by-token. Delegates to the Taichi
    kernel when the runtime is available; otherwise falls back to the
    pure-PyTorch reference (:func:`torch_recurrent_gdn2`). Output is always
    returned as a contiguous tensor so downstream ``torch.compile`` /
    ``opcheck`` callers see a stable memory layout.

    Args:
        q: Queries ``[B, T, H, K]``.
        k: Keys ``[B, T, H, K]``.
        v: Values ``[B, T, H, V]``.
        g: Log-decay gate ``[B, T, H, K]``.
        b: Channel-wise erase gate ``[B, T, H, K]``.
        w: Channel-wise write gate ``[B, T, H, V]``.
        initial_state: Optional initial recurrent state ``[B, H, K, V]``.
        output_final_state: If ``True``, return the final state.
        use_qk_l2norm_in_kernel: Apply L2 normalization to q/k.

    Returns:
        out: Output tensor ``[B, T, H, V]``.
        state: Final recurrent state ``[B, H, K, V]`` if
            ``output_final_state`` is ``True``, else ``None``.
    """
    if _is_available():
        from qwendopamine.kernels.taichi.gdn2_api import (
            recurrent_taichi_gdn2 as _fn,
        )

        out, state = _fn(
            q,
            k,
            v,
            g,
            b,
            w,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if state is not None and state.data_ptr() == (
            initial_state.data_ptr() if initial_state is not None else 0
        ):
            state = state.clone()
        return (out.contiguous(), state) if out is not None else (q.new_empty(0), state)
    out, state = torch_recurrent_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    if state is not None and state.data_ptr() == (
        initial_state.data_ptr() if initial_state is not None else 0
    ):
        state = state.clone()
    return (out.contiguous(), state) if out is not None else (q.new_empty(0), state)
