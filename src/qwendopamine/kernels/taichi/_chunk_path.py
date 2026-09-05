"""Chunkwise Taichi GDN-2 path.

Extracted from :mod:`gdn2_api` for size.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from qwendopamine.kernels.taichi import gdn2_kernels as _kernels
from qwendopamine.kernels.taichi import runtime as _rt


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

    The forward output is the **chunkwise** result (WY transform, not
    per-token replay). The per-token ``states`` are captured via a
    single extra per-token pass that writes only into ``states``, not
    into ``out``. This keeps the chunkwise Taichi kernel as the
    production forward engine and avoids the ``O(T)`` forward
    replay that the previous implementation did.
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
            # S_{end} = gamma[-1] * S_{start} + kbar^T @ delta
            # (rank-1 term carries no decay factor; ``kbar`` is already
            # scaled by ``gamma`` and ``delta`` already absorbs the WY
            # rotation, so applying ``gamma`` again would double-count
            # the channel-wise decay on the rank-1 contribution).
            rank_one = torch.matmul(kbar.transpose(-1, -2), delta)
            state = gamma[:, :, -1:, :].transpose(-1, -2) * state + rank_one

    # Capture per-token states for the Taichi adjoint. The chunkwise
    # forward only produces the per-chunk entry/exit states, so we run
    # a single per-token pass that writes ONLY into ``states`` (not
    # into ``out``). This preserves the chunkwise Taichi forward
    # output above and lets the bwd path reuse the per-token VJP we
    # already verified against the canonical reference.
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
            # The per-token kernel also writes ``y`` (the output). We
            # discard it: the chunkwise output above is the production
            # value, this kernel call is state-only.
            y=out.new_empty(B, H, V, dtype=torch.float32, device=q.device),
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
    because the Taichi path templates ``C`` at compile time. The
    returned output is the chunkwise result, not a per-token replay.
    Backward consumes the per-token states captured during the
    forward and runs the per-token adjoint kernel in reverse, so
    training flows end-to-end through the public API even though the
    gradient kernel itself is per-token (the forward numerics are
    still Taichi).
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
]
