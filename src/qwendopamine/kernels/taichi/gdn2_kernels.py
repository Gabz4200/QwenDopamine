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
    T = qs.shape[2]
    B = qs.shape[0]
    H = qs.shape[1]
    K = qs.shape[-1]
    V = vs.shape[-1]
    device = qs.device
    # Per-step scratch buffers (one at a time, reused across T iterations).
    dq_t = torch.zeros((B, H, K), dtype=torch.float32, device=device)
    dk_t = torch.zeros((B, H, K), dtype=torch.float32, device=device)
    dv_t = torch.zeros((B, H, V), dtype=torch.float32, device=device)
    da_t = torch.zeros((B, H, K), dtype=torch.float32, device=device)
    db_t = torch.zeros((B, H, K), dtype=torch.float32, device=device)
    dw_t = torch.zeros((B, H, V), dtype=torch.float32, device=device)
    scratch = torch.zeros((B, H, K, V), dtype=torch.float32, device=device)
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
        dq[:, :, t, :].copy_(dq_t)
        dk[:, :, t, :].copy_(dk_t)
        dv[:, :, t, :].copy_(dv_t)
        da[:, :, t, :].copy_(da_t)
        db[:, :, t, :].copy_(db_t)
        dw[:, :, t, :].copy_(dw_t)
    # dstate_in holds the gradient w.r.t. the chunk's initial state.
    dstate_in.copy_(dstate_next)


# Scratch buffers for the chunkwise kernel. We allocate them lazily at
# the first invocation of a given (C, K, V) combination and reuse them
# across calls. Each buffer lives in Taichi's memory space and is indexed
# from inside the kernel. Python-side callers do not interact with these
# directly; the launch wrappers pass them as kernel arguments.
_SCRATCH: Any = {}


def _get_chunk_scratch(
    C: int, K: int, V: int, device: torch.device | str
) -> dict[str, Any]:
    key = (C, K, V, str(device))
    if key in _SCRATCH:
        result: dict[str, Any] = _SCRATCH[key]
        return result
    _rt.require()  # ensure taichi is initialised
    _SCRATCH[key] = {
        "gamma": torch.zeros((C, K), dtype=torch.float32, device=device),
        "kbar": torch.zeros((C, K), dtype=torch.float32, device=device),
        "ebar": torch.zeros((C, K), dtype=torch.float32, device=device),
        "z": torch.zeros((C, V), dtype=torch.float32, device=device),
        "Y": torch.zeros((C, K), dtype=torch.float32, device=device),
        "U": torch.zeros((C, V), dtype=torch.float32, device=device),
        "delta": torch.zeros((C, V), dtype=torch.float32, device=device),
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
        for j in range(K):
            g_acc = rt.f32(0.0)
            for i in range(C):
                g_acc = g_acc + g_log[i, j]
                gamma[i, j] = rt.exp(g_acc)

        for i in range(C):
            for j in range(K):
                kbar[i, j] = k[i, j] / rt.max(gamma[i, j], 1e-12)
                ebar[i, j] = gamma[i, j] * (b[i, j] * k[i, j])
            for j in range(V):
                z[i, j] = w[i, j] * v[i, j]

        for i in range(C):
            for j in range(K):
                Y[i, j] = ebar[i, j]
            for j in range(V):
                U[i, j] = z[i, j]
            for prev in range(i):
                t_y = rt.f32(0.0)
                for j in range(K):
                    t_y = t_y + ebar[i, j] * kbar[prev, j]
                for j in range(K):
                    Y[i, j] = Y[i, j] - t_y * Y[prev, j]
                for j in range(V):
                    U[i, j] = U[i, j] - t_y * U[prev, j]

        for i in range(C):
            for j in range(V):
                acc = rt.f32(0.0)
                for kk in range(K):
                    acc = acc + Y[i, kk] * state_in[kk, j]
                delta[i, j] = U[i, j] - acc

        for kk in range(K):
            for j in range(V):
                acc = rt.f32(0.0)
                for i in range(C):
                    acc = acc + kbar[i, kk] * delta[i, j]
                state_out[kk, j] = gamma[C - 1, kk] * (state_in[kk, j] + acc)

        for i in range(C):
            for j in range(V):
                acc = rt.f32(0.0)
                for kk in range(K):
                    acc = acc + (gamma[i, kk] * q[i, kk]) * state_in[kk, j]
                for s in range(i + 1):
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


def _build_chunk_intra_token_parallel_kernel() -> Any:
    r"""Build the intra-chunk causal score matrices Aqk and Akk.

    For each token position ``i`` within a chunk of size ``C``:

    (Aqk)_{r,i} = 1_{i<=r} (gamma[r] * q[r])^T kbar[i]       (C, C)
    (Akk)_{r,i} = 1_{i<=r} (gamma[r] * k[r])^T kbar[i]       (C, C)

    where ``kbar = k / gamma``.  Both are causal: entries with ``i > r``
    are zero.  The kernel is embarrassingly parallel across tokens and
    across (batch, head) — it writes every element independently.
    """
    rt = _rt.require()

    @rt.kernel  # pyrefly: ignore[untyped-function-decorator]
    def chunk_intra_token_parallel(  # pyrefly: ignore[unannotated-return]
        q: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        k: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        gamma: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        kbar: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        Aqk: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        Akk: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        K: rt.i32,
        C: rt.i32,
    ):
        for i in range(C):
            for j in range(C):
                if j <= i:
                    aqk = rt.f32(0.0)
                    akk = rt.f32(0.0)
                    for kk in range(K):
                        qg = gamma[i, kk] * q[i, kk]
                        aqk = aqk + qg * kbar[j, kk]
                        kg = gamma[i, kk] * k[i, kk]
                        akk = akk + kg * kbar[j, kk]
                    Aqk[i, j] = aqk
                    Akk[i, j] = akk
                else:
                    Aqk[i, j] = rt.f32(0.0)
                    Akk[i, j] = rt.f32(0.0)

    return chunk_intra_token_parallel


def launch_chunk_intra_token_parallel(
    q_chunk: torch.Tensor,
    k_chunk: torch.Tensor,
    gamma: torch.Tensor,
    kbar: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
) -> None:
    """Build causal intra-chunk score matrices Aqk and Akk for one (batch, head).

    All inputs are ``[C, K]`` (single batch, single head, single chunk);
    outputs are ``[C, C]``.
    """
    K = int(q_chunk.shape[-1])
    C = int(q_chunk.shape[-2])
    kernel = _build_chunk_intra_token_parallel_kernel()
    kernel(
        q_chunk.contiguous(),
        k_chunk.contiguous(),
        gamma.contiguous(),
        kbar.contiguous(),
        Aqk,
        Akk,
        K,
        C,
    )


def _build_wy_fast_kernel() -> Any:
    r"""Recompute the WY auxiliaries (w, u, qg, kg) from the solved matrix ``A``.

    Mirrors FLA ``recompute_w_u_fwd_gdn2_kernel``.  Given the solved
    WY lower-triangular inverse ``A = (I + tril(T, -1))^{-1}`` of shape
    ``[C, C]`` (one chunk, one head), and the per-token intermediates of
    that chunk, produce:

    * ``w_out`` — the WY write auxiliary ``A @ ebar``  (``[C, K]``)
    * ``u``     — the WY erase auxiliary ``A @ z``     (``[C, V]``)
    * ``qg``    — query gate scaled by gamma         (``[C, K]``)
    * ``kg``    — key scaled by tail-decay           (``[C, K]``)

    where ``ebar = gamma * (b * k)``, ``z = w_gate * v``, and
    ``gamma[t,k] = exp(sum_{s<=t} g_log[s,k])``.  The ``qg``/``kg``
    outputs are optional (FLA's ``STORE_QG``/``STORE_KG`` heuristics).
    """
    rt = _rt.require()

    @rt.kernel  # pyrefly: ignore[untyped-function-decorator]
    def wy_fast(  # pyrefly: ignore[unannotated-return]
        A: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        k: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        q: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        b: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        w_gate: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        v: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        gamma: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        g_log: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        w_out: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        u: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        qg: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        kg: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        K: rt.i32,
        V: rt.i32,
        C: rt.i32,
    ):
        for i in range(C):
            for vv in range(V):
                acc = rt.f32(0.0)
                for j in range(C):
                    acc = acc + A[i, j] * (w_gate[j, vv] * v[j, vv])
                u[i, vv] = acc

        for i in range(C):
            for kk in range(K):
                acc = rt.f32(0.0)
                for j in range(C):
                    ebar_jk = gamma[j, kk] * (b[j, kk] * k[j, kk])
                    acc = acc + A[i, j] * ebar_jk
                w_out[i, kk] = acc

        for i in range(C):
            for kk in range(K):
                qg[i, kk] = gamma[i, kk] * q[i, kk]

        for i in range(C):
            for kk in range(K):
                gm_last = gamma[C - 1, kk]
                gm_cur = gamma[i, kk]
                safe_gm = rt.max(gm_cur, rt.f32(1e-12))
                kg[i, kk] = (gm_last / safe_gm) * k[i, kk]

    return wy_fast


def launch_wy_fast(
    A: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
    g_log: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute WY auxiliaries (w, u, qg, kg) from the solved WY matrix ``A``.

    All inputs are ``[C, ...]`` (single chunk, single head).  Returns
    ``(w, u, qg, kg)``.
    """
    C = int(A.shape[-1])
    K = int(k.shape[-1])
    V = int(v.shape[-1])
    w_out = torch.zeros(C, K, dtype=torch.float32, device=A.device)
    u = torch.zeros(C, V, dtype=torch.float32, device=A.device)
    qg = torch.zeros(C, K, dtype=torch.float32, device=A.device)
    kg = torch.zeros(C, K, dtype=torch.float32, device=A.device)
    kernel = _build_wy_fast_kernel()
    kernel(
        A.contiguous(),
        k.contiguous(),
        q.contiguous(),
        b.contiguous(),
        w_gate.contiguous(),
        v.contiguous(),
        gamma.contiguous(),
        g_log.contiguous() if g_log is not None else gamma.contiguous(),
        w_out,
        u,
        qg,
        kg,
        K,
        V,
        C,
    )
    return w_out, u, qg, kg


def launch_naive_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Naive O(T*K*V) GDN-2 per-token recurrence.

    Mirrors FLA's ``naive.py``: a pure-Python loop over the existing
    :func:`launch_recurrent_step` Taichi kernel. Each call advances the
    ``[K, V]`` state by one token. ``g`` is the per-token log-decay
    (not cumulative); ``alpha = exp(g[t])`` is computed inside the
    recurrent-step kernel.

    All inputs are ``[T, K]`` or ``[T, V]`` (single batch, single head).
    ``q`` must already be scaled by ``K**-0.5``. Returns
    ``(output, final_state)``.
    """
    _rt.require()
    T = int(q.shape[0])
    K = int(q.shape[-1])
    V = int(v.shape[-1])
    out = torch.zeros(T, V, dtype=torch.float32, device=q.device)
    state = torch.zeros(1, 1, K, V, dtype=torch.float32, device=q.device)
    next_state = torch.zeros(1, 1, K, V, dtype=torch.float32, device=q.device)
    y_scratch = torch.zeros(1, 1, V, dtype=torch.float32, device=q.device)
    alphas = torch.exp(g.float())

    for t in range(T):
        launch_recurrent_step(
            state=state,
            q=q[t].unsqueeze(0).unsqueeze(0).contiguous(),
            k=k[t].unsqueeze(0).unsqueeze(0).contiguous(),
            v=v[t].unsqueeze(0).unsqueeze(0).contiguous(),
            a=alphas[t].unsqueeze(0).unsqueeze(0).contiguous(),
            b=b[t].unsqueeze(0).unsqueeze(0).contiguous(),
            w=w[t].unsqueeze(0).unsqueeze(0).contiguous(),
            next_state=next_state,
            y=y_scratch,
        )
        out[t] = y_scratch[0, 0]
        state, next_state = next_state, state

    return out, state


def _build_chunk_fwd_pipeline_kernel() -> Any:
    r"""Full chunkwise forward pipeline (FLA chunk_fwd equivalent).

    Combines intra-chunk scoring, WY solve, inter-chunk state propagation,
    and output composition into a single fused kernel per (batch, head).
    """
    rt = _rt.require()

    @rt.kernel  # pyrefly: ignore[untyped-function-decorator]
    def chunk_fwd_pipeline(  # pyrefly: ignore[unannotated-return]
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
        Aqk: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        Akk: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        A: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        Y: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        U: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        delta: rt.types.ndarray(),  # type: ignore[invalid-annotation]
        scale: rt.f32,
        K: rt.i32,
        V: rt.i32,
        C: rt.i32,
    ):
        for kk in range(K):
            acc = rt.f32(0.0)
            for t in range(C):
                acc = acc + g_log[t, kk]
                gamma[t, kk] = rt.exp(acc)

        for i in range(C):
            for kk in range(K):
                gm = rt.max(gamma[i, kk], rt.f32(1e-12))
                kbar[i, kk] = k[i, kk] / gm
                ebar[i, kk] = gamma[i, kk] * (b[i, kk] * k[i, kk])
            for vv in range(V):
                z[i, vv] = w[i, vv] * v[i, vv]

        for i in range(C):
            for j in range(C):
                if j <= i:
                    aqk = rt.f32(0.0)
                    for kk in range(K):
                        aqk = aqk + (gamma[i, kk] * q[i, kk]) * kbar[j, kk]
                    Aqk[i, j] = aqk
                    akk = rt.f32(0.0)
                    for kk in range(K):
                        akk = akk + (gamma[i, kk] * k[i, kk]) * kbar[j, kk]
                    Akk[i, j] = akk
                else:
                    Aqk[i, j] = rt.f32(0.0)
                    Akk[i, j] = rt.f32(0.0)

        for i in range(C):
            for kk in range(K):
                Y[i, kk] = ebar[i, kk]
            for vv in range(V):
                U[i, vv] = z[i, vv]
            for prev in range(i):
                t_coeff = rt.f32(0.0)
                for kk in range(K):
                    t_coeff = t_coeff + ebar[i, kk] * kbar[prev, kk]
                for kk in range(K):
                    Y[i, kk] = Y[i, kk] - t_coeff * Y[prev, kk]
                for vv in range(V):
                    U[i, vv] = U[i, vv] - t_coeff * U[prev, vv]
            for j in range(C):
                if j < i:
                    t_val = rt.f32(0.0)
                    for kk in range(K):
                        t_val = t_val + ebar[i, kk] * kbar[j, kk]
                    A[i, j] = t_val
                elif j == i:
                    A[i, j] = rt.f32(1.0)
                else:
                    A[i, j] = rt.f32(0.0)

        for i in range(C):
            for vv in range(V):
                acc = rt.f32(0.0)
                for kk in range(K):
                    acc = acc + Y[i, kk] * state_in[kk, vv]
                delta[i, vv] = U[i, vv] - acc

        for kk in range(K):
            for vv in range(V):
                acc = rt.f32(0.0)
                for i in range(C):
                    acc = acc + kbar[i, kk] * delta[i, vv]
                state_out[kk, vv] = gamma[C - 1, kk] * (state_in[kk, vv] + acc)

        for i in range(C):
            for vv in range(V):
                acc = rt.f32(0.0)
                for kk in range(K):
                    acc = acc + (gamma[i, kk] * q[i, kk]) * state_in[kk, vv]
                for s in range(i + 1):
                    acc = acc + Aqk[i, s] * delta[s, vv]
                out[i, vv] = acc * scale

    return chunk_fwd_pipeline


def launch_chunk_fwd_pipeline(
    q_chunk: torch.Tensor,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    g_chunk: torch.Tensor,
    b_chunk: torch.Tensor,
    w_chunk: torch.Tensor,
    state_in: torch.Tensor,
    out: torch.Tensor,
    state_out: torch.Tensor,
    scratch: dict[str, torch.Tensor],
    scale: float,
) -> None:
    """Full chunkwise forward pipeline for one (batch, head).

    Orchestrates intra-chunk scoring, WY solve, inter-chunk state
    propagation, and output composition in a single fused kernel.
    """
    K = int(q_chunk.shape[-1])
    V = int(v_chunk.shape[-1])
    C = int(q_chunk.shape[-2])
    kernel = _build_chunk_fwd_pipeline_kernel()
    kernel(
        q_chunk.contiguous(),
        k_chunk.contiguous(),
        v_chunk.contiguous(),
        g_chunk.contiguous(),
        b_chunk.contiguous(),
        w_chunk.contiguous(),
        state_in.contiguous(),
        out,
        state_out,
        scratch["gamma"],
        scratch["kbar"],
        scratch["ebar"],
        scratch["z"],
        scratch["Aqk"],
        scratch["Akk"],
        scratch["A"],
        scratch["Y"],
        scratch["U"],
        scratch["delta"],
        scale,
        K,
        V,
        C,
    )


def _get_chunk_scratch_full(
    C: int, K: int, V: int, device: torch.device | str
) -> dict[str, Any]:
    """Lazily allocate scratch buffers for the full pipeline kernel."""
    key = (C, K, V, str(device), "full")
    if key in _SCRATCH:
        result: dict[str, Any] = _SCRATCH[key]
        return result
    _rt.require()
    _SCRATCH[key] = {
        "gamma": torch.zeros((C, K), dtype=torch.float32, device=device),
        "kbar": torch.zeros((C, K), dtype=torch.float32, device=device),
        "ebar": torch.zeros((C, K), dtype=torch.float32, device=device),
        "z": torch.zeros((C, V), dtype=torch.float32, device=device),
        "Aqk": torch.zeros((C, C), dtype=torch.float32, device=device),
        "Akk": torch.zeros((C, C), dtype=torch.float32, device=device),
        "A": torch.zeros((C, C), dtype=torch.float32, device=device),
        "Y": torch.zeros((C, K), dtype=torch.float32, device=device),
        "U": torch.zeros((C, V), dtype=torch.float32, device=device),
        "delta": torch.zeros((C, V), dtype=torch.float32, device=device),
    }
    result2: dict[str, Any] = _SCRATCH[key]
    return result2


__all__ = [
    "launch_chunk_bwd_per_bh",
    "launch_chunk_fwd_per_bh",
    "launch_chunk_fwd_pipeline",
    "launch_chunk_intra_token_parallel",
    "launch_naive_gdn2",
    "launch_recurrent_step",
    "launch_recurrent_step_bwd",
    "launch_wy_fast",
]
