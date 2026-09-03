"""High-level API for the Taichi-accelerated GDN-2 kernels.

The functions in this module are drop-in replacements for the previous
``torch_chunk_gdn2`` / ``torch_recurrent_gdn2`` / ``triton.chunk_gdn2`` /
``triton.fused_recurrent_gdn2`` entry points. They use the same calling
convention (PyTorch tensors in BTHD layout) so the rest of the
``GatedDeltaNet2`` block does not need to know whether the underlying
engine is Taichi, Triton, or pure PyTorch.

Both the recurrent and the chunkwise paths are wrapped in
``torch.autograd.Function`` so training (backward) flows end-to-end.
The recurrent path stores every per-token state and per-token
activation and replays the token loop in reverse during ``backward``;
its per-step adjoint is implemented by
:func:`_kernels.launch_recurrent_step_bwd`. The chunkwise path keeps
the Taichi forward (production engine) and re-runs the equivalent
torch reference in reverse to obtain per-input gradients; that torch
reference matches the Taichi numerics because the public functions
operate on identical inputs and identical mathematical algorithm.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from qwendopamine.kernels.taichi import gdn2_kernels as _kernels
from qwendopamine.kernels.taichi import runtime as _rt


def _normalize_qk(
    q: torch.Tensor, k: torch.Tensor, apply: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optional L2-normalisation on Q and K. Matched to the legacy flags."""
    if not apply:
        return q, k
    q = F.normalize(q, p=2, dim=-1, eps=1e-6)
    k = F.normalize(k, p=2, dim=-1, eps=1e-6)
    return q, k


# ---------------------------------------------------------------------------
# Recurrent autograd wrapper
# ---------------------------------------------------------------------------


class _RecurrentTaichiGdn2Function(torch.autograd.Function):
    """Autograd wrapper around the Taichi recurrent GDN-2 forward.

    Forward is a token-by-token loop; we save every per-token state and
    per-token activation so backward can replay the loop in reverse.
    The per-token adjoint is implemented by
    :func:`_kernels.launch_recurrent_step_bwd`. ``q`` and ``k`` reach
    the kernel already L2-normalised and pre-scaled; gradients to the
    caller's raw ``q``/``k`` flow through the autograd-aware
    :func:`F.normalize` and the ``d_k**-0.5`` scalar.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, final_state, saved = _recurrent_taichi_gdn2_inner(
            q,
            k,
            v,
            g,
            b,
            w,
            initial_state,
        )
        ctx.save_for_backward(*saved)
        ctx.has_initial_state = initial_state is not None
        return out, final_state

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_out: torch.Tensor | None,
        grad_final_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_out is None:
            return (None,) * 7
        (states, qs, ks, vs, alphas, bs, ws) = ctx.saved_tensors
        T = qs.shape[2]
        if grad_final_state is None:
            dstate = torch.zeros_like(states[0])
        else:
            dstate = grad_final_state.float().clone()
        dq = torch.zeros_like(qs)
        dk = torch.zeros_like(ks)
        dv = torch.zeros_like(vs)
        dalpha = torch.zeros_like(alphas)
        db = torch.zeros_like(bs)
        dw = torch.zeros_like(ws)
        grad_out = grad_out.float()
        # grad_out is [B, T, H, V]; rearrange to [B, H, T, V] to match qs/ks.
        grad_out = rearrange(grad_out, "b t h d -> b h t d").contiguous()
        # Per-step scratch buffers (contiguous, kernel-compatible).
        dq_t = dq.new_zeros((dq.shape[0], dq.shape[1], dq.shape[-1]))
        dk_t = dk.new_zeros((dk.shape[0], dk.shape[1], dk.shape[-1]))
        dv_t = dv.new_zeros((dv.shape[0], dv.shape[1], dv.shape[-1]))
        da_t = dalpha.new_zeros((dalpha.shape[0], dalpha.shape[1], dalpha.shape[-1]))
        db_t = db.new_zeros((db.shape[0], db.shape[1], db.shape[-1]))
        dw_t = dw.new_zeros((dw.shape[0], dw.shape[1], dw.shape[-1]))
        scratch = states[0].new_zeros(states[0].shape)
        for t in reversed(range(T)):
            state_in = states[t]
            state_out = states[t + 1]
            _kernels.launch_recurrent_step_bwd(
                state_in=state_in,
                state_out=state_out,
                q=qs[:, :, t, :].contiguous(),
                k=ks[:, :, t, :].contiguous(),
                v=vs[:, :, t, :].contiguous(),
                a=alphas[:, :, t, :].contiguous(),
                b=bs[:, :, t, :].contiguous(),
                w=ws[:, :, t, :].contiguous(),
                dy=grad_out[:, :, t, :].contiguous(),
                dstate_out=dstate,
                dstate_in=scratch,
                dq=dq_t,
                dk=dk_t,
                dv=dv_t,
                da=da_t,
                db=db_t,
                dw=dw_t,
            )
            # Roll the state buffer.
            dstate, scratch = scratch, dstate
            # Accumulate per-step grads into the [B,H,T,...] outputs.
            dq[:, :, t, :].copy_(dq_t)
            dk[:, :, t, :].copy_(dk_t)
            dv[:, :, t, :].copy_(dv_t)
            dalpha[:, :, t, :].copy_(da_t)
            db[:, :, t, :].copy_(db_t)
            dw[:, :, t, :].copy_(dw_t)

        # d_a -> d_g via a = exp(g)
        dg = dalpha * alphas
        # The kernel saw q pre-multiplied by the d_k**-0.5 scale (the
        # scaling is applied inside this Function), so the kernel's
        # dq is dL/d(q_scaled). Convert to dL/dq_in by multiplying by
        # the same scale; PyTorch then propagates the F.normalize VJP
        # on the caller's original q.
        scale = qs.shape[-1] ** -0.5
        dq = dq * scale
        # Re-arrange all input grads back to [B, T, H, K/V] so they
        # match the caller's original tensor shapes.
        dq = rearrange(dq, "b h t d -> b t h d").contiguous()
        dk = rearrange(dk, "b h t d -> b t h d").contiguous()
        dv = rearrange(dv, "b h t d -> b t h d").contiguous()
        dg = rearrange(dg, "b h t d -> b t h d").contiguous()
        db = rearrange(db, "b h t d -> b t h d").contiguous()
        dw = rearrange(dw, "b h t d -> b t h d").contiguous()
        d_initial_state: torch.Tensor | None = dstate if ctx.has_initial_state else None
        return dq, dk, dv, dg, db, dw, d_initial_state


def _recurrent_taichi_gdn2_inner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Forward token loop. Returns output, final state, and saved tensors.

    Saved tensors: ``(states, qs, ks, vs, alphas, bs, ws)`` where
    ``states`` is shape ``[T+1, B, H, K, V]`` and the rest are
    ``[B, H, T, ...]``.
    """
    _rt.require()
    B, T, H, K = q.shape
    V = v.shape[-1]

    # q, k arrive already L2-normalised. Apply the d_k**-0.5 scale here
    # so the kernel sees the production-scale inputs.
    q = q.float() * (K**-0.5)
    k = k.float()
    v = v.float()
    g = g.float()
    b = b.float()
    w = w.float()

    q = rearrange(q, "b t h d -> b h t d").contiguous()
    k = rearrange(k, "b t h d -> b h t d").contiguous()
    v = rearrange(v, "b t h d -> b h t d").contiguous()
    g = rearrange(g, "b t h d -> b h t d").contiguous()
    b = rearrange(b, "b t h d -> b h t d").contiguous()
    w = rearrange(w, "b t h d -> b h t d").contiguous()

    if initial_state is None:
        state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.float().clone()

    states = torch.empty(T + 1, B, H, K, V, dtype=torch.float32, device=q.device)
    states[0].copy_(state)
    alphas = torch.exp(g).contiguous()
    next_state = torch.empty_like(state)
    outputs = torch.empty(B, H, T, V, dtype=torch.float32, device=q.device)
    y_scratch = torch.empty(B, H, V, dtype=torch.float32, device=q.device)

    for t in range(T):
        _kernels.launch_recurrent_step(
            state=state,
            q=q[:, :, t, :].contiguous(),
            k=k[:, :, t, :].contiguous(),
            v=v[:, :, t, :].contiguous(),
            a=alphas[:, :, t, :].contiguous(),
            b=b[:, :, t, :].contiguous(),
            w=w[:, :, t, :].contiguous(),
            next_state=next_state,
            y=y_scratch,
        )
        outputs[:, :, t, :].copy_(y_scratch)
        states[t + 1].copy_(next_state)
        state, next_state = next_state, state

    out = rearrange(outputs, "b h t d -> b t h d")
    return out, state, (states, q, k, v, alphas, b, w)


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
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure Taichi recurrent GDN-2 forward + backward.

    Training (backward) flows through the per-token Taichi adjoint
    kernel :func:`_kernels.launch_recurrent_step_bwd`, which is the
    token-by-token VJP of paper Eq. 10. The L2-normalisation and
    ``d_k**-0.5`` scale are autograd-friendly (``F.normalize`` is
    autograd-aware; the scale is a scalar multiplication).
    """
    _rt.require()
    # Cast to float32 without breaking the autograd graph: ``.to`` is
    # a no-op when the dtype already matches, so requires_grad and the
    # gradient chain are preserved.
    if use_qk_l2norm_in_kernel:
        q_in = F.normalize(q.to(torch.float32), p=2, dim=-1, eps=1e-6)
        k_in = F.normalize(k.to(torch.float32), p=2, dim=-1, eps=1e-6)
    else:
        q_in = q.to(torch.float32)
        k_in = k.to(torch.float32)
    v_in = v.to(torch.float32)
    g_in = g.to(torch.float32)
    b_in = b.to(torch.float32)
    w_in = w.to(torch.float32)
    if initial_state is None:
        init = None
    else:
        init = initial_state.to(torch.float32)
    out, final_state = _RecurrentTaichiGdn2Function.apply(
        q_in,
        k_in,
        v_in,
        g_in,
        b_in,
        w_in,
        init,
    )
    if not output_final_state:
        final_state = None
    else:
        if final_state is not None and final_state.dtype != q.dtype:
            final_state = final_state.to(q.dtype)
    return out.to(q.dtype), final_state


# ---------------------------------------------------------------------------
# Chunkwise autograd wrapper
# ---------------------------------------------------------------------------


class _ChunkTaichiGdn2Function(torch.autograd.Function):
    """Autograd wrapper around the chunkwise Taichi path.

    Forward runs the WY-based chunkwise algorithm in Taichi (production
    engine) and saves per-token states. Backward replays the per-token
    adjoint kernel ``launch_recurrent_step_bwd`` in reverse over the
    saved states, which is the mathematically correct chunkwise
    adjoint (the chunkwise forward is a re-arrangement of the same
    recurrence, paper Eq. 23-24 vs Eq. 10).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        initial_state: torch.Tensor | None,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, final_state, saved = _chunk_taichi_gdn2_inner(
            q,
            k,
            v,
            g,
            b,
            w,
            initial_state,
            chunk_size,
        )
        (states, qs, ks, vs, alphas, bs, ws) = saved
        ctx.save_for_backward(
            states,
            qs,
            ks,
            vs,
            alphas,
            bs,
            ws,
            q,
            k,
            v,
            g,
            b,
            w,
        )
        ctx.has_initial_state = initial_state is not None
        ctx.chunk_size = chunk_size
        return out, final_state

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_out: torch.Tensor | None,
        grad_final_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_out is None:
            return (None,) * 8
        (
            states,
            qs,
            ks,
            vs,
            alphas,
            bs,
            ws,
            _q,
            _k,
            _v,
            _g,
            _b,
            _w,
        ) = ctx.saved_tensors
        B = qs.shape[0]
        H = qs.shape[1]
        K = qs.shape[-1]
        V = vs.shape[-1]
        # grad_out is [B, T, H, V]; rearrange to [B, H, T, V].
        grad_out_bh = rearrange(grad_out, "b t h d -> b h t d").contiguous()
        dq = torch.zeros_like(qs)
        dk = torch.zeros_like(ks)
        dv = torch.zeros_like(vs)
        da = torch.zeros_like(alphas)
        db = torch.zeros_like(bs)
        dw = torch.zeros_like(ws)
        if grad_final_state is None:
            dstate_next = torch.zeros(
                B,
                H,
                K,
                V,
                dtype=torch.float32,
                device=qs.device,
            )
        else:
            dstate_next = grad_final_state.float().clone()
        dstate_in = torch.zeros_like(dstate_next)
        _kernels.launch_chunk_bwd_per_bh(
            states=states,
            qs=qs,
            ks=ks,
            vs=vs,
            alphas=alphas,
            bs=bs,
            ws=ws,
            grad_out=grad_out_bh,
            dstate_next=dstate_next,
            dstate_in=dstate_in,
            dq=dq,
            dk=dk,
            dv=dv,
            da=da,
            db=db,
            dw=dw,
        )
        # da is w.r.t. alpha = exp(g); d_g = d_alpha * alpha.
        dg = da * alphas
        # Re-arrange the [B, H, T, ...] grads back to the [B, T, H, ...]
        # layout the caller expects.
        dq = rearrange(dq, "b h t d -> b t h d").contiguous()
        dk = rearrange(dk, "b h t d -> b t h d").contiguous()
        dv = rearrange(dv, "b h t d -> b t h d").contiguous()
        dg = rearrange(dg, "b h t d -> b t h d").contiguous()
        db = rearrange(db, "b h t d -> b t h d").contiguous()
        dw = rearrange(dw, "b h t d -> b t h d").contiguous()
        d_initial_state: torch.Tensor | None = (
            dstate_in if ctx.has_initial_state else None
        )
        return dq, dk, dv, dg, db, dw, d_initial_state, None


def _chunk_taichi_gdn2_inner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Chunkwise forward. Returns output, final state, and saved tensors.

    Saved tensors: ``(states, qs, ks, vs, alphas, bs, ws)`` where
    ``states`` is shape ``[T+1, B, H, K, V]`` (per-token states for
    the Taichi adjoint to consume in reverse order). The remaining
    saved tensors are the per-token q/k/v/alpha/b/w used by the
    per-token VJP ``launch_recurrent_step_bwd``.
    """
    _rt.require()
    B, T, H, K = q.shape
    V = v.shape[-1]

    q = q.float() * (K**-0.5)
    k = k.float()
    v = v.float()
    g = g.float()
    b = b.float()
    w = w.float()

    q = rearrange(q, "b t h d -> b h t d").contiguous()
    k = rearrange(k, "b t h d -> b h t d").contiguous()
    v = rearrange(v, "b t h d -> b h t d").contiguous()
    g = rearrange(g, "b t h d -> b h t d").contiguous()
    b = rearrange(b, "b t h d -> b h t d").contiguous()
    w = rearrange(w, "b t h d -> b h t d").contiguous()

    if initial_state is None:
        state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.float().clone()

    out = torch.empty(B, H, T, V, dtype=torch.float32, device=q.device)
    states = torch.empty(T + 1, B, H, K, V, dtype=torch.float32, device=q.device)
    states[0].copy_(state)
    alphas = torch.exp(g).contiguous()
    NT = (T + chunk_size - 1) // chunk_size

    for chunk_idx in range(NT):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, T)
        C = end - start
        if C == chunk_size:
            q_c = q[:, :, start:end].contiguous()
            k_c = k[:, :, start:end].contiguous()
            v_c = v[:, :, start:end].contiguous()
            g_c = g[:, :, start:end].contiguous()
            b_c = b[:, :, start:end].contiguous()
            w_c = w[:, :, start:end].contiguous()
            new_state = torch.empty_like(state)
            scratch = _kernels._get_chunk_scratch(C, K, V)
            for bh in range(B * H):
                bbh = bh // H
                hbh = bh % H
                _kernels.launch_chunk_fwd_per_bh(
                    q_c[bbh, hbh],
                    k_c[bbh, hbh],
                    v_c[bbh, hbh],
                    g_c[bbh, hbh],
                    b_c[bbh, hbh],
                    w_c[bbh, hbh],
                    state[bbh, hbh],
                    out[bbh, hbh, start:end],
                    new_state[bbh, hbh],
                    scratch["gamma"],
                    scratch["kbar"],
                    scratch["ebar"],
                    scratch["z"],
                    scratch["Y"],
                    scratch["U"],
                    scratch["delta"],
                    1.0,
                )
            state = new_state
        else:
            from qwendopamine.models.gdn2.recurrence.chunk import (
                compute_gdn2_intra_chunk_scores,
                compute_gdn2_wy_coefficients,
            )

            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]
            g_c = g[:, :, start:end]
            b_c = b[:, :, start:end]
            w_c = w[:, :, start:end]
            gamma = torch.exp(torch.cumsum(g_c, dim=2))
            gam_safe = gamma.clamp_min(1e-12)
            kbar = k_c / gam_safe
            ebar = gamma * (b_c * k_c)
            z = w_c * v_c
            y, u = compute_gdn2_wy_coefficients(kbar, ebar, z, device=q.device)
            delta = u - torch.matmul(y, state)
            q_gamma = gamma * q_c
            out_inter = torch.matmul(q_gamma, state)
            aqk = compute_gdn2_intra_chunk_scores(q_c, gamma, kbar)
            out[:, :, start:end] = out_inter + torch.matmul(aqk, delta)
            state = gamma[:, :, -1:, :].transpose(-1, -2) * (
                state + torch.matmul(kbar.transpose(-1, -2), delta)
            )

    # Capture per-token states for the Taichi adjoint. The chunkwise
    # forward only produces the per-chunk entry/exit states, so we
    # replay the per-token recurrent kernel here. This adds T launches
    # but keeps the chunkwise Taichi kernel as the production engine
    # and lets the bwd path reuse the per-token VJP we already
    # verified against the canonical reference.
    for t in range(T):
        _kernels.launch_recurrent_step(
            state=states[t].contiguous(),
            q=q[:, :, t, :].contiguous(),
            k=k[:, :, t, :].contiguous(),
            v=v[:, :, t, :].contiguous(),
            a=alphas[:, :, t, :].contiguous(),
            b=b[:, :, t, :].contiguous(),
            w=w[:, :, t, :].contiguous(),
            next_state=states[t + 1].contiguous(),
            y=out[:, :, t, :].contiguous(),
        )

    out = rearrange(out, "b h t d -> b t h d")
    return out, state, (states, q, k, v, alphas, b, w)


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
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure Taichi chunkwise GDN-2 forward + backward.

    Forward dispatches each full chunk to a per-(batch, head) Taichi
    kernel. Tail chunks that don't fill ``chunk_size`` run in PyTorch
    because the Taichi path templates ``C`` at compile time. Backward
    defers to the differentiable torch reference for gradient
    computation, so training flows end-to-end through the public API
    even though the gradient kernel itself is pure torch (the forward
    numerics are still Taichi).
    """
    _rt.require()
    if use_qk_l2norm_in_kernel:
        q_in = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        k_in = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
    else:
        q_in = q.float()
        k_in = k.float()
    v_in = v.float()
    g_in = g.float()
    b_in = b.float()
    w_in = w.float()
    if initial_state is None:
        init = None
    else:
        init = initial_state.float()
    out, final_state = _ChunkTaichiGdn2Function.apply(
        q_in,
        k_in,
        v_in,
        g_in,
        b_in,
        w_in,
        init,
        chunk_size,
    )
    if not output_final_state:
        final_state = None
    else:
        if final_state is not None and final_state.dtype != q.dtype:
            final_state = final_state.to(q.dtype)
    return out.to(q.dtype), final_state


__all__ = [
    "chunk_taichi_gdn2",
    "recurrent_taichi_gdn2",
]
