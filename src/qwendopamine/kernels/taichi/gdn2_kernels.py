"""Taichi kernels implementing the Gated DeltaNet-2 (GDN-2) recurrence.

The kernels below cover the forward single-token recurrence and the
adjoint backward that pairs with the recurrent forward. The forward
matches the equation:

    S_dec[k,v]  = alpha[k] * S[k,v]
    k_erased[k] = b[k] * k[k]
    v_ret       = sum_k S_dec[k,v] * k_erased[k]
    v_write[v]  = w[v] * v[v] - v_ret[v]
    S_next      = S_dec + outer(k, v_write)
    y[v]        = sum_k S_next[k,v] * q[k]

For one token step the backward (VJP) given upstream grad ``dy[v]`` and
``dstate_out[k,v]`` (gradient w.r.t. the post-update state, propagated
from the next step) is:

    dS_next[k,v] = q[k] * dy[v] + dstate_out[k,v]
    d_v_write[v] = sum_k k[k] * dS_next[k,v]
    d_v_ret[v]   = -d_v_write[v]
    d_w[v]       = v[v] * d_v_write[v]
    d_v[v]       = w[v] * d_v_write[v]
    d_k_write[k] = sum_v v_write[v] * dS_next[k,v]
    d_b[k]       = k[k] * sum_v S_dec[k,v] * d_v_ret[v]
    d_k_erase[k] = b[k] * sum_v S_dec[k,v] * d_v_ret[v]
    d_k[k]       = d_k_write[k] + d_k_erase[k]
    dS_dec[k,v]  = dS_next[k,v] + k_erased[k] * d_v_ret[v]
    d_state_in[k,v] = alpha[k] * dS_dec[k,v]
    d_alpha[k]  = sum_v S_in[k,v] * dS_dec[k,v]
    d_q[k]      = sum_v S_next[k,v] * dy[v]

When stacked across the time axis, the saved ``S_t`` at every step makes
the recurrence backpropagation a clean linear sweep. The backward kernel
below recomputes the small forward intermediates (``S_dec``, ``v_write``)
from the saved inputs to avoid storing extra scratch.
"""

from typing import Any

import torch

from qwendopamine.kernels.taichi import runtime as _rt

ti: Any = _rt.ti  # type: ignore[assignment]


def _build_recurrent_step_kernel() -> Any:
    rt = _rt.require()

    @rt.kernel  # pyrefly: ignore[untyped-function-decorator]
    def recurrent_step(  # pyrefly: ignore[unannotated-return]
        state: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        q: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        k: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        v: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        a: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        b: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        w: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        next_state: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        y: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        K: rt.i32,
        V: rt.i32,
    ):
        for i_b, i_h in rt.ndrange(state.shape[0], state.shape[1]):
            for i_v in range(V):
                for i_k in range(K):
                    next_state[i_b, i_h, i_k, i_v] = (
                        a[i_b, i_h, i_k] * state[i_b, i_h, i_k, i_v]
                    )
            for i_v in range(V):
                read = rt.f32(0.0)
                for i_k in range(K):
                    read = (
                        read
                        + (b[i_b, i_h, i_k] * k[i_b, i_h, i_k])
                        * next_state[i_b, i_h, i_k, i_v]
                    )
                write_term = w[i_b, i_h, i_v] * v[i_b, i_h, i_v] - read
                for i_k in range(K):
                    next_state[i_b, i_h, i_k, i_v] = (
                        next_state[i_b, i_h, i_k, i_v] + k[i_b, i_h, i_k] * write_term
                    )
                y_val = rt.f32(0.0)
                for i_k in range(K):
                    y_val = y_val + q[i_b, i_h, i_k] * next_state[i_b, i_h, i_k, i_v]
                y[i_b, i_h, i_v] = y_val

    return recurrent_step


def launch_recurrent_step(state, q, k, v, a, b, w, next_state, y) -> None:
    """Apply the GDN-2 single-token recurrence on ``state`` -> ``next_state``."""
    K = int(q.shape[-1])
    V = int(v.shape[-1])
    kernel = _build_recurrent_step_kernel()
    kernel(state, q, k, v, a, b, w, next_state, y, K, V)


def _build_recurrent_step_bwd_kernel() -> Any:
    rt = _rt.require()

    @rt.kernel  # pyrefly: ignore[untyped-function-decorator]
    def recurrent_step_bwd(  # pyrefly: ignore[unannotated-return]
        state_in: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        state_out: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        q: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        k: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        v: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        a: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        b: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        w: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        dy: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        dstate_out: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        dstate_in: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        dq: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        dk: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        dv: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        da: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        db: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        dw: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        K: rt.i32,
        V: rt.i32,
    ):
        r"""Per-token VJP matching :func:`_build_recurrent_step_kernel`.

        Inputs: forward tensors (``state_in``, ``state_out``, ``q/k/v``,
        ``a/b/w``) plus upstream ``dy`` and ``dstate_out`` (the gradient
        w.r.t. ``state_out`` propagated from the next step). Outputs:
        ``dstate_in`` (flow to the previous step), ``dq``/``dk``/``dv``/
        ``da``/``db``/``dw`` (per-input gradients).
        """
        for i_b, i_h in rt.ndrange(state_in.shape[0], state_in.shape[1]):
            # Pass 1: compute d_v_write[v] and d_v_ret[v], and the
            # dS_dec buffer in ``dstate_in`` (overwriting the caller-
            # provided zero buffer).
            for i_v in range(V):
                d_v_write = rt.f32(0.0)
                for i_k in range(K):
                    d_v_write = d_v_write + k[i_b, i_h, i_k] * (
                        q[i_b, i_h, i_k] * dy[i_b, i_h, i_v]
                        + dstate_out[i_b, i_h, i_k, i_v]
                    )
                dw[i_b, i_h, i_v] = v[i_b, i_h, i_v] * d_v_write
                dv[i_b, i_h, i_v] = w[i_b, i_h, i_v] * d_v_write
                d_v_ret = -d_v_write
                # dS_dec[k,v] = dS_next[k,v] + k_erased[k] * d_v_ret[v]
                for i_k in range(K):
                    k_erased = b[i_b, i_h, i_k] * k[i_b, i_h, i_k]
                    dstate_in[i_b, i_h, i_k, i_v] = (
                        q[i_b, i_h, i_k] * dy[i_b, i_h, i_v]
                        + dstate_out[i_b, i_h, i_k, i_v]
                        + k_erased * d_v_ret
                    )

            # Pass 2: per-(k) reductions.
            for i_k in range(K):
                # dq[k] = sum_v state_out[k,v] * dy[v]
                dq_val = rt.f32(0.0)
                for i_v in range(V):
                    dq_val = dq_val + state_out[i_b, i_h, i_k, i_v] * dy[i_b, i_h, i_v]
                dq[i_b, i_h, i_k] = dq_val

                # d_k_write[k] = sum_v v_write[v] * dS_next[k,v]
                d_k_write = rt.f32(0.0)
                # d_b[k] = k[k] * sum_v S_dec[k,v] * d_v_ret[v]
                d_b_val = rt.f32(0.0)
                # d_k_erase[k] = b[k] * sum_v S_dec[k,v] * d_v_ret[v]
                d_k_erase = rt.f32(0.0)
                for i_v in range(V):
                    # v_write[v] = w[v] * v[v] - sum_{k'} b[k'] * k[k'] * a[k'] * S[k',v]
                    v_write = w[i_b, i_h, i_v] * v[i_b, i_h, i_v]
                    s_dec_kv = a[i_b, i_h, i_k] * state_in[i_b, i_h, i_k, i_v]
                    for i_kk in range(K):
                        v_write = v_write - (b[i_b, i_h, i_kk] * k[i_b, i_h, i_kk]) * (
                            a[i_b, i_h, i_kk] * state_in[i_b, i_h, i_kk, i_v]
                        )
                    dS_next_kv = (
                        q[i_b, i_h, i_k] * dy[i_b, i_h, i_v]
                        + dstate_out[i_b, i_h, i_k, i_v]
                    )
                    d_k_write = d_k_write + v_write * dS_next_kv
                    # d_v_ret[v] = -d_v_write[v] = -sum_{k'} k[k'] * dS_next[k',v]
                    d_v_ret = rt.f32(0.0)
                    for i_kk in range(K):
                        d_v_ret = d_v_ret - k[i_b, i_h, i_kk] * (
                            q[i_b, i_h, i_kk] * dy[i_b, i_h, i_v]
                            + dstate_out[i_b, i_h, i_kk, i_v]
                        )
                    d_b_val = d_b_val + k[i_b, i_h, i_k] * s_dec_kv * d_v_ret
                    d_k_erase = d_k_erase + b[i_b, i_h, i_k] * s_dec_kv * d_v_ret
                dk[i_b, i_h, i_k] = d_k_write + d_k_erase
                db[i_b, i_h, i_k] = d_b_val

                # d_a[k] = sum_v state_in[k,v] * dS_dec[k,v]
                da_val = rt.f32(0.0)
                for i_v in range(V):
                    da_val = (
                        da_val
                        + state_in[i_b, i_h, i_k, i_v] * dstate_in[i_b, i_h, i_k, i_v]
                    )
                da[i_b, i_h, i_k] = da_val

                # d_state_in[k,v] = a[k] * dS_dec[k,v]
                for i_v in range(V):
                    dstate_in[i_b, i_h, i_k, i_v] = (
                        a[i_b, i_h, i_k] * dstate_in[i_b, i_h, i_k, i_v]
                    )

    return recurrent_step_bwd


def launch_recurrent_step_bwd(
    state_in,
    state_out,
    q,
    k,
    v,
    a,
    b,
    w,
    dy,
    dstate_out,
    dstate_in,
    dq,
    dk,
    dv,
    da,
    db,
    dw,
) -> None:
    """Run the per-token adjoint of :func:`launch_recurrent_step`.

    All inputs and outputs are ``[B, H, ...]``-shaped ndarrays.
    ``dstate_out`` is the gradient w.r.t. the state at the end of the
    step (from the next token's backward); ``dy`` is the gradient w.r.t.
    the per-token output ``y``. The function overwrites ``dstate_in``
    with the gradient w.r.t. the state at the start of the step (for
    the previous token's backward) and writes per-input gradients into
    ``dq/dk/dv/da/db/dw``. All output buffers must be pre-zeroed.
    """
    K = int(q.shape[-1])
    V = int(v.shape[-1])
    kernel = _build_recurrent_step_bwd_kernel()
    kernel(
        state_in,
        state_out,
        q,
        k,
        v,
        a,
        b,
        w,
        dy,
        dstate_out,
        dstate_in,
        dq,
        dk,
        dv,
        da,
        db,
        dw,
        K,
        V,
    )


def launch_chunk_bwd_per_bh(
    states,
    qs,
    ks,
    vs,
    alphas,
    bs,
    ws,
    grad_out,
    dstate_next,
    dstate_in,
    dq,
    dk,
    dv,
    da,
    db,
    dw,
) -> None:
    """Run the chunkwise adjoint by replaying the per-token VJP in reverse.

    The Taichi recurrent-step bwd kernel is the token-by-token adjoint
    of paper Eq. 10 (verified by the canonical validation tests).
    Replaying it in reverse over the saved per-token states is the
    mathematically correct chunkwise adjoint: the chunkwise forward is
    a re-arrangement of the same recurrence (paper Eq. 23-24 vs Eq. 10),
    so the per-token adjoint also reproduces the chunkwise gradient to
    fp32 accumulation noise.

    All input tensors are ``[B, H, T, ...]``-shaped (``states`` is
    ``[T+1, B, H, K, V]``). All output gradient buffers must be
    pre-zeroed.
    """
    import numpy as np

    T = qs.shape[2]
    B = qs.shape[0]
    H = qs.shape[1]
    K = qs.shape[-1]
    V = vs.shape[-1]
    # Per-step scratch buffers (one at a time, reused across T iterations).
    dq_t = np.zeros((B, H, K), dtype=np.float32)
    dk_t = np.zeros((B, H, K), dtype=np.float32)
    dv_t = np.zeros((B, H, V), dtype=np.float32)
    da_t = np.zeros((B, H, K), dtype=np.float32)
    db_t = np.zeros((B, H, K), dtype=np.float32)
    dw_t = np.zeros((B, H, V), dtype=np.float32)
    scratch = np.zeros((B, H, K, V), dtype=np.float32)
    for t in reversed(range(T)):
        state_in = states[t]
        state_out = states[t + 1]
        launch_recurrent_step_bwd(
            state_in=state_in,
            state_out=state_out,
            q=qs[:, :, t, :].contiguous(),
            k=ks[:, :, t, :].contiguous(),
            v=vs[:, :, t, :].contiguous(),
            a=alphas[:, :, t, :].contiguous(),
            b=bs[:, :, t, :].contiguous(),
            w=ws[:, :, t, :].contiguous(),
            dy=grad_out[:, :, t, :].contiguous(),
            dstate_out=dstate_next,
            dstate_in=scratch,
            dq=dq_t,
            dk=dk_t,
            dv=dv_t,
            da=da_t,
            db=db_t,
            dw=dw_t,
        )
        # Roll the dstate buffer.
        dstate_next, scratch = scratch, dstate_next
        # Accumulate per-step grads into the [B,H,T,...] outputs.
        dq[:, :, t, :].copy_(torch.from_numpy(dq_t).to(dq.device))
        dk[:, :, t, :].copy_(torch.from_numpy(dk_t).to(dk.device))
        dv[:, :, t, :].copy_(torch.from_numpy(dv_t).to(dv.device))
        da[:, :, t, :].copy_(torch.from_numpy(da_t).to(da.device))
        db[:, :, t, :].copy_(torch.from_numpy(db_t).to(db.device))
        dw[:, :, t, :].copy_(torch.from_numpy(dw_t).to(dw.device))
    # dstate_in holds the gradient w.r.t. the chunk's initial state.
    dstate_in.copy_(dstate_next)


# Scratch buffers for the chunkwise kernel. We allocate them lazily at
# the first invocation of a given (C, K, V) combination and reuse them
# across calls. Each buffer lives in Taichi's memory space and is indexed
# from inside the kernel. Python-side callers do not interact with these
# directly; the launch wrappers pass them as kernel arguments.
_SCRATCH: Any = {}


def _get_chunk_scratch(C: int, K: int, V: int) -> dict[str, Any]:
    key = (C, K, V)
    if key in _SCRATCH:
        result: dict[str, Any] = _SCRATCH[key]
        return result
    import numpy as np

    _rt.require()  # ensure taichi is initialised
    _SCRATCH[key] = {
        "gamma": np.zeros((C, K), dtype=np.float32),
        "kbar": np.zeros((C, K), dtype=np.float32),
        "ebar": np.zeros((C, K), dtype=np.float32),
        "z": np.zeros((C, V), dtype=np.float32),
        "Y": np.zeros((C, K), dtype=np.float32),
        "U": np.zeros((C, V), dtype=np.float32),
        "delta": np.zeros((C, V), dtype=np.float32),
    }
    result2: dict[str, Any] = _SCRATCH[key]
    return result2


def _build_chunk_fwd_per_bh_kernel() -> Any:
    rt = _rt.require()

    @rt.kernel  # pyrefly: ignore[untyped-function-decorator]
    def chunk_fwd_bh(  # pyrefly: ignore[unannotated-return]
        q: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        k: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        v: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        g_log: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        b: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        w: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        state_in: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        out: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        state_out: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        gamma: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        kbar: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        ebar: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        z: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        Y: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        U: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        delta: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        scale: rt.f32,
        K: rt.i32,
        V: rt.i32,
        C: rt.template(),  # pyrefly: ignore[invalid-annotation]
    ):
        for i in ti.static(range(C)):
            g_acc = rt.f32(0.0)
            for j in range(K):
                g_acc = g_acc + g_log[i, j]
                gamma[i, j] = rt.exp(g_acc)

        for i in ti.static(range(C)):
            for j in range(K):
                kbar[i, j] = k[i, j] / rt.max(gamma[i, j], 1e-12)
                ebar[i, j] = gamma[i, j] * (b[i, j] * k[i, j])
            for j in range(V):
                z[i, j] = w[i, j] * v[i, j]

        for i in ti.static(range(C)):
            for j in range(K):
                Y[i, j] = ebar[i, j]
            for j in range(V):
                U[i, j] = z[i, j]
            for prev in ti.static(range(i)):
                t_y = rt.f32(0.0)
                for j in range(K):
                    t_y = t_y + ebar[i, j] * kbar[prev, j]
                for j in range(K):
                    Y[i, j] = Y[i, j] - t_y * Y[prev, j]
                for j in range(V):
                    U[i, j] = U[i, j] - t_y * U[prev, j]

        for i in ti.static(range(C)):
            for j in range(V):
                acc = rt.f32(0.0)
                for kk in range(K):
                    acc = acc + Y[i, kk] * state_in[kk, j]
                delta[i, j] = U[i, j] - acc

        for kk in range(K):
            for j in range(V):
                acc = rt.f32(0.0)
                for i in ti.static(range(C)):
                    acc = acc + kbar[i, kk] * delta[i, j]
                state_out[kk, j] = gamma[C - 1, kk] * (state_in[kk, j] + acc)

        for i in ti.static(range(C)):
            for j in range(V):
                acc = rt.f32(0.0)
                for kk in range(K):
                    acc = acc + (gamma[i, kk] * q[i, kk]) * state_in[kk, j]
                for s in ti.static(range(i + 1)):
                    a_qk = rt.f32(0.0)
                    for kk in range(K):
                        a_qk = a_qk + (gamma[i, kk] * q[i, kk]) * kbar[s, kk]
                    acc = acc + a_qk * delta[s, j]
                out[i, j] = acc * scale

    return chunk_fwd_bh


def launch_chunk_fwd_per_bh(
    q_chunk,
    k_chunk,
    v_chunk,
    g_chunk,
    b_chunk,
    w_chunk,
    state_in,
    out,
    state_out,
    gamma,
    kbar,
    ebar,
    z,
    Y,
    U,
    delta,
    scale,
) -> None:
    """Run one chunk of the WY forward kernel for a single (batch, head)."""
    K = int(q_chunk.shape[-1])
    V = int(v_chunk.shape[-1])
    C = int(q_chunk.shape[-2])
    kernel = _build_chunk_fwd_per_bh_kernel()
    kernel(
        q_chunk,
        k_chunk,
        v_chunk,
        g_chunk,
        b_chunk,
        w_chunk,
        state_in,
        out,
        state_out,
        gamma,
        kbar,
        ebar,
        z,
        Y,
        U,
        delta,
        scale,
        K,
        V,
        C,
    )


__all__ = [
    "launch_chunk_bwd_per_bh",
    "launch_chunk_fwd_per_bh",
    "launch_recurrent_step",
    "launch_recurrent_step_bwd",
]
