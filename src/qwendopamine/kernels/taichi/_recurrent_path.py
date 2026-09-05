"""Recurrent Taichi GDN-2 path.

Extracted from :mod:`gdn2_api` for size.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from qwendopamine.kernels.taichi import gdn2_kernels as _kernels
from qwendopamine.kernels.taichi import runtime as _rt


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
