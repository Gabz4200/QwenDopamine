"""Taichi-accelerated kernels for the Reinforced Delta memory core.

The :class:`DeltaMemoryCore` from :mod:`qwendopamine.models.reinforced.delta`
maintains a matrix-valued fast-weight state ``S_t`` in ``R^{d x d}`` and
applies the coupled write/erase update

    S_{t+1} = (1 - omega_E * E_t) * S_t + omega_W * (e_t k_t^T)
    o_t     = S_{t+1}^T q_t

where ``E_t`` is the channel-wise erase gate, ``e_t = v_t - S_t k_t`` is
the delta residual, and ``omega_{W,E}`` are the advantage-modulated
scalar plasticity factors. This module provides single Taichi kernels
for both the forward state update and the per-token adjoint.

The FiLM-modulated readout is computed in PyTorch afterwards (it
depends on the reward statistics, not on the recurrent state) so the
kernels stay focused on the matrix-state hot path.

Shape contract (single-step, per-batch):
    state          [B, D, D]
    k, v           [B, D]
    omega_w_eff    [B, D]    (per-channel effective gate, = omega_w * write)
    omega_e_eff    [B, D]    (per-channel effective gate, = omega_e * erase)
    next_state     [B, D, D]

Shape contract (backward, per-batch):
    dstate         [B, D, D]
    dk, dv         [B, D]
    d_omega_w_eff  [B, D]
    d_omega_e_eff  [B, D]
"""

from typing import Any

import torch

from qwendopamine.kernels.taichi import runtime as _rt

ti = _rt.ti  # type: ignore[assignment]


def _make_effective_gate(omega: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Contract the public ``[B, 1]`` or ``[B]`` omega with the
    channel-wise ``gate`` of shape ``[B, D]`` into the internal
    ``[B, D]`` per-channel effective gate.

    Both the forward and backward kernels receive this 2D buffer.
    Doing the conversion exactly once at the public boundary keeps
    the kernel access pattern unambiguous.
    """
    if omega.dim() == 1:
        omega = omega.unsqueeze(-1)  # [B, 1]
    if omega.dim() != 2 or omega.shape[-1] != 1:
        raise ValueError(f"omega must be [B] or [B, 1]; got shape {tuple(omega.shape)}")
    return (omega * gate).contiguous()


def _build_delta_core_step_kernel() -> Any:
    rt = _rt.require()

    @rt.kernel  # pyrefly: ignore[untyped-function-decorator]
    def delta_core_step(  # pyrefly: ignore[unannotated-return]
        state: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        k: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        v: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        omega_w_eff: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        omega_e_eff: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        next_state: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        D: rt.i32,
    ):
        # Column-wise update: each (i_b, i_d) thread owns the
        # next_state[i_b, i_d, *] row. The torch reference treats the
        # state as ``S[b, d, k]`` (row=d, col=k) with read
        # ``e[d] = v[d] - sum_kk S[d, kk] * k[kk]`` and rank-1 write
        # ``S_next[d, k] = (1 - omega_e[d]) * S[d, k] + omega_w[d] * e[d] * k[k]``.
        # omega_w_eff / omega_e_eff are 2D [B, D] (per-channel).
        for i_b, i_d in rt.ndrange(state.shape[0], D):
            read = rt.f32(0.0)
            for kk in range(D):
                read = read + state[i_b, i_d, kk] * k[i_b, kk]
            e_val = v[i_b, i_d] - read
            w_term = omega_w_eff[i_b, i_d] * e_val
            oe_d = omega_e_eff[i_b, i_d]
            for kk in range(D):
                next_state[i_b, i_d, kk] = (1.0 - oe_d) * state[
                    i_b, i_d, kk
                ] + w_term * k[i_b, kk]

    return delta_core_step


def launch_delta_core_step(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    next_state: torch.Tensor,
) -> None:
    """Apply one Reinforced Delta state update: state -> next_state.

    ``omega_w`` and ``omega_e`` are per-batch scalar weights; they may be
    ``[B]`` or ``[B, 1]``. ``erase`` and ``write`` are channel-wise gates
    of shape ``[B, D]``. The function contracts the public omega
    shape into the internal 2D per-channel buffer expected by the
    kernel.
    """
    omega_w_eff = _make_effective_gate(omega_w, write)
    omega_e_eff = _make_effective_gate(omega_e, erase)
    D = int(k.shape[-1])
    kernel = _build_delta_core_step_kernel()
    kernel(
        state.contiguous(),
        k.contiguous(),
        v.contiguous(),
        omega_w_eff,
        omega_e_eff,
        next_state.contiguous() if next_state is not None else None,
        D,
    )


def _build_delta_core_step_bwd_kernel() -> Any:
    rt = _rt.require()

    @rt.kernel  # pyrefly: ignore[untyped-function-decorator]
    def delta_core_step_bwd(  # pyrefly: ignore[unannotated-return]
        state: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        k: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        v: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        omega_w_eff: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        omega_e_eff: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        dnext_state: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        dstate: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        dk: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        dv: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        d_omega_w_eff: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        d_omega_e_eff: rt.types.ndarray(),  # pyrefly: ignore[invalid-annotation]
        D: rt.i32,
    ):
        r"""Per-token VJP matching :func:`_build_delta_core_step_kernel`.

        Forward (state layout is ``S[b, d, k]``):
            e[d] = v[d] - sum_{kk} S[d, kk] * k[kk]
            w_term[d] = omega_w_eff[d] * e[d]
            S_next[d, k] = (1 - omega_e_eff[d]) * S[d, k] + w_term[d] * k[k]

        Let ``r[d] = sum_kk k[kk] * dS_next[d, kk]``. Then:
            d_v[d] = omega_w_eff[d] * r[d]
            d_omega_w_eff[d] = e[d] * r[d]
            d_omega_e_eff[d] = -sum_k S[d, k] * dS_next[d, k]
            dstate[d, k] = (1 - omega_e_eff[d]) * dS_next[d, k]
                           - k[k] * omega_w_eff[d] * r[d]
            dk[k] = sum_d dS_next[d, k] * w_term[d]
                    - sum_d S[d, k] * omega_w_eff[d] * r[d]
        """
        for i_b in rt.ndrange(state.shape[0]):
            # Pass 1: per-d reductions.
            for i_d in range(D):
                read = rt.f32(0.0)
                for i_kk in range(D):
                    read = read + state[i_b, i_d, i_kk] * k[i_b, i_kk]
                e_val = v[i_b, i_d] - read
                r_d = rt.f32(0.0)
                for i_kk in range(D):
                    r_d = r_d + k[i_b, i_kk] * dnext_state[i_b, i_d, i_kk]
                w_term_d = omega_w_eff[i_b, i_d] * e_val
                d_omega_w_eff[i_b, i_d] = e_val * r_d
                dv[i_b, i_d] = omega_w_eff[i_b, i_d] * r_d
                oe_d = 1.0 - omega_e_eff[i_b, i_d]
                d_oe_val = rt.f32(0.0)
                for i_k in range(D):
                    d_oe_val = (
                        d_oe_val - state[i_b, i_d, i_k] * dnext_state[i_b, i_d, i_k]
                    )
                d_omega_e_eff[i_b, i_d] = d_oe_val
                # dstate[d, k] = (1 - oe_e[d]) * dS_next[d, k] - k[k] * ow_e[d] * r[d]
                for i_k in range(D):
                    dstate[i_b, i_d, i_k] = (
                        oe_d * dnext_state[i_b, i_d, i_k]
                        - k[i_b, i_k] * omega_w_eff[i_b, i_d] * r_d
                    )
                # Stash w_term_d for Pass 2 via the dv buffer is
                # wrong (dv is the output). Instead, recompute w_term_d
                # in Pass 2 (cheap; both read and e_val depend only on
                # state, k, v which are immutable across passes).

            # Pass 2: per-k reduction for dk.
            for i_k in range(D):
                dk_val = rt.f32(0.0)
                for i_d in range(D):
                    # Recompute e_val[d] and r[d] (Pass 1 didn't store
                    # them; recomputation keeps the kernel stateless
                    # across the two passes at the cost of O(D^2)
                    # extra work, which is fine for correctness).
                    e_val_d = v[i_b, i_d]
                    for i_kk in range(D):
                        e_val_d = e_val_d - state[i_b, i_d, i_kk] * k[i_b, i_kk]
                    r_d = rt.f32(0.0)
                    for i_kk in range(D):
                        r_d = r_d + k[i_b, i_kk] * dnext_state[i_b, i_d, i_kk]
                    w_term_d = omega_w_eff[i_b, i_d] * e_val_d
                    # Write path: dS_next[d, k] * w_term[d]
                    dk_val = dk_val + dnext_state[i_b, i_d, i_k] * w_term_d
                    # Read path: -S[d, k] * omega_w_eff[d] * r[d]
                    dk_val = dk_val - state[i_b, i_d, i_k] * omega_w_eff[i_b, i_d] * r_d
                dk[i_b, i_k] = dk_val

    return delta_core_step_bwd


def launch_delta_core_step_bwd(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w_eff: torch.Tensor,
    omega_e_eff: torch.Tensor,
    dnext_state: torch.Tensor,
    dstate: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    d_omega_w_eff: torch.Tensor,
    d_omega_e_eff: torch.Tensor,
) -> None:
    """Per-token adjoint of :func:`launch_delta_core_step`.

    ``omega_w_eff`` and ``omega_e_eff`` are 2D ``[B, D]`` per-channel
    effective gates. The function overwrites ``dstate`` with the
    gradient w.r.t. the prior state (for the previous token's
    backward) and writes per-input gradients into ``dk``, ``dv``,
    ``d_omega_w_eff``, ``d_omega_e_eff``. All output buffers must be
    pre-zeroed.
    """
    D = int(k.shape[-1])
    kernel = _build_delta_core_step_bwd_kernel()
    kernel(
        state,
        k,
        v,
        omega_w_eff,
        omega_e_eff,
        dnext_state,
        dstate,
        dk,
        dv,
        d_omega_w_eff,
        d_omega_e_eff,
        D,
    )


class _DeltaCoreStepFunction(torch.autograd.Function):
    """Autograd wrapper around :func:`launch_delta_core_step`.

    Contract the public ``[B, 1]`` omega and the per-channel
    ``[B, D]`` write/erase gates into the internal ``[B, D]``
    ``omega_w_eff`` / ``omega_e_eff`` buffers before the kernel
    launch. The autograd graph then distributes gradients back to the
    original inputs via the product rule (no division by zero).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        state,
        k,
        v,
        omega_w,
        omega_e,
        write,
        erase,
        next_state,
    ):
        _rt.require()
        omega_w_eff = _make_effective_gate(omega_w, write)
        omega_e_eff = _make_effective_gate(omega_e, erase)
        out = torch.empty_like(state)
        launch_delta_core_step(
            state,
            k,
            v,
            omega_w,
            omega_e,
            erase,
            write,
            out,
        )
        if next_state is not None and next_state.data_ptr() != out.data_ptr():
            next_state.copy_(out)
        ctx.save_for_backward(
            state,
            k,
            v,
            omega_w_eff,
            omega_e_eff,
            write,
            erase,
            omega_w,
            omega_e,
        )
        return out

    @staticmethod
    def backward(ctx, grad_next_state):  # type: ignore[override]
        if grad_next_state is None:
            return (None,) * 8
        (
            state,
            k,
            v,
            omega_w_eff,
            omega_e_eff,
            write,
            erase,
            omega_w,
            omega_e,
        ) = ctx.saved_tensors
        dstate = torch.zeros_like(state)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        d_omega_w_eff = torch.zeros_like(omega_w_eff)
        d_omega_e_eff = torch.zeros_like(omega_e_eff)
        launch_delta_core_step_bwd(
            state.contiguous(),
            k.contiguous(),
            v.contiguous(),
            omega_w_eff.contiguous(),
            omega_e_eff.contiguous(),
            grad_next_state.contiguous(),
            dstate,
            dk,
            dv,
            d_omega_w_eff,
            d_omega_e_eff,
        )
        # Product-rule distribution from the effective 2D gates back
        # to the per-batch scalar ``omega_w`` and per-channel
        # ``write``. Avoid division (numerically fragile when the
        # gate is zero).
        #   omega_w_eff = omega_w * write  (per-channel)
        #   d_omega_w   = sum_d d_omega_w_eff[d] * write[d]  (per-batch scalar)
        #   d_write     = d_omega_w_eff * omega_w             (per-channel)
        if omega_w.dim() == 1:
            ow_for_dist = omega_w.unsqueeze(-1)  # [B, 1]
        else:
            ow_for_dist = omega_w
        if omega_e.dim() == 1:
            oe_for_dist = omega_e.unsqueeze(-1)
        else:
            oe_for_dist = omega_e
        d_omega_w = (d_omega_w_eff * write).sum(dim=-1, keepdim=True)
        d_omega_e = (d_omega_e_eff * erase).sum(dim=-1, keepdim=True)
        d_write = d_omega_w_eff * ow_for_dist
        d_erase = d_omega_e_eff * oe_for_dist
        if omega_w.dim() == 1:
            d_omega_w = d_omega_w.squeeze(-1)
        if omega_e.dim() == 1:
            d_omega_e = d_omega_e.squeeze(-1)
        return dstate, dk, dv, d_omega_w, d_omega_e, d_write, d_erase, None


def delta_core_step_out(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
    next_state: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Reinforced Delta state update.

    Returns the new state ``next_state`` with gradients flowing
    through the per-token Taichi adjoint kernel.
    """
    return _DeltaCoreStepFunction.apply(
        state,
        k,
        v,
        omega_w,
        omega_e,
        write,
        erase,
        next_state,
    )


def launch_chunk_bwd_per_bh(
    states,
    ks,
    vs,
    omega_w_effs,
    omega_e_effs,
    dstate_next,
    dstate_in,
    dks,
    dvs,
    d_omega_w_effs,
    d_omega_e_effs,
) -> None:
    """Run the per-token RewardNet VJP in reverse over a chunk's saved states.

    The Taichi per-step adjoint kernel ``launch_delta_core_step_bwd``
    is the token-by-token VJP of the RewardNet update. Replaying it in
    reverse over the saved per-token states is the mathematically
    correct chunkwise adjoint (the chunkwise forward is a re-arrangement
    of the same per-token recurrence, paper Eq. 29 and Appendix A).

    ``omega_w_effs`` and ``omega_e_effs`` are ``[B, T, D]`` per-step
    effective gates. All other inputs are ``[B, T, D]`` or
    ``[T+1, B, D, D]`` for ``states``. All output gradient buffers
    must be pre-zeroed.
    """
    import numpy as np

    T = ks.shape[1]
    B = ks.shape[0]
    D = ks.shape[-1]
    # Per-step scratch buffers (one at a time, reused across T iterations).
    dk_t = np.zeros((B, D), dtype=np.float32)
    dv_t = np.zeros((B, D), dtype=np.float32)
    d_omega_w_eff_t = np.zeros((B, D), dtype=np.float32)
    d_omega_e_eff_t = np.zeros((B, D), dtype=np.float32)
    scratch = np.zeros((B, D, D), dtype=np.float32)
    for t in reversed(range(T)):
        state_in = states[t]
        launch_delta_core_step_bwd(
            state=state_in,
            k=ks[:, t, :].contiguous(),
            v=vs[:, t, :].contiguous(),
            omega_w_eff=omega_w_effs[:, t, :].contiguous(),
            omega_e_eff=omega_e_effs[:, t, :].contiguous(),
            dnext_state=dstate_next,
            dstate=scratch,
            dk=dk_t,
            dv=dv_t,
            d_omega_w_eff=d_omega_w_eff_t,
            d_omega_e_eff=d_omega_e_eff_t,
        )
        # Roll the dstate buffer.
        dstate_next, scratch = scratch, dstate_next
        # Accumulate per-step grads into the [B, T, D] outputs.
        dks[:, t, :].copy_(torch.from_numpy(dk_t).to(dks.device))
        dvs[:, t, :].copy_(torch.from_numpy(dv_t).to(dvs.device))
        d_omega_w_effs[:, t, :].copy_(
            torch.from_numpy(d_omega_w_eff_t).to(d_omega_w_effs.device),
        )
        d_omega_e_effs[:, t, :].copy_(
            torch.from_numpy(d_omega_e_eff_t).to(d_omega_e_effs.device),
        )
    dstate_in.copy_(dstate_next)


class _ChunkwiseDeltaCoreStepFunction(torch.autograd.Function):
    """Autograd wrapper around the chunkwise RewardNet Taichi path.

    Forward loops :func:`launch_delta_core_step` over the time axis
    and saves per-token states plus the per-step effective
    ``omega_w_eff = omega_w * write`` and
    ``omega_e_eff = omega_e * erase``. Backward replays the per-token
    VJP via :func:`launch_chunk_bwd_per_bh` in reverse order, which is
    the mathematically correct chunkwise adjoint (the chunkwise
    forward is the same per-token recurrence applied in order).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        state0,
        k,
        v,
        omega_w,
        omega_e,
        write,
        erase,
        next_state,
    ):
        _rt.require()
        B, T, D = k.shape
        ow_eff = omega_w * write  # [B, T, D]
        oe_eff = omega_e * erase  # [B, T, D]
        states = torch.empty(
            T + 1,
            B,
            D,
            D,
            dtype=torch.float32,
            device=k.device,
        )
        states[0].copy_(state0.float())
        next_scratch = torch.empty_like(states[0])
        for t in range(T):
            launch_delta_core_step(
                state=states[t].contiguous(),
                k=k[:, t, :].contiguous(),
                v=v[:, t, :].contiguous(),
                omega_w=ow_eff[:, t, :].contiguous(),
                omega_e=oe_eff[:, t, :].contiguous(),
                erase=erase[:, t, :].contiguous(),
                write=write[:, t, :].contiguous(),
                next_state=next_scratch,
            )
            states[t + 1].copy_(next_scratch)
        if next_state is not None and next_state.data_ptr() != states[T].data_ptr():
            next_state.copy_(states[T])
        ctx.save_for_backward(
            states,
            k,
            v,
            ow_eff,
            oe_eff,
            write,
            erase,
            omega_w,
            omega_e,
        )
        return states[T].to(state0.dtype)

    @staticmethod
    def backward(ctx, grad_next_state):  # type: ignore[override]
        if grad_next_state is None:
            return (None,) * 8
        (
            states,
            k,
            v,
            ow_eff,
            oe_eff,
            write,
            erase,
            omega_w,
            omega_e,
        ) = ctx.saved_tensors
        dks = torch.zeros_like(k)
        dvs = torch.zeros_like(v)
        d_ow_eff = torch.zeros_like(ow_eff)
        d_oe_eff = torch.zeros_like(oe_eff)
        dstate_next = grad_next_state.float().clone()
        dstate_in = torch.zeros_like(dstate_next)
        launch_chunk_bwd_per_bh(
            states=states,
            ks=k,
            vs=v,
            omega_w_effs=ow_eff,
            omega_e_effs=oe_eff,
            dstate_next=dstate_next,
            dstate_in=dstate_in,
            dks=dks,
            dvs=dvs,
            d_omega_w_effs=d_ow_eff,
            d_omega_e_effs=d_oe_eff,
        )
        # Product-rule distribution. Per-step effective gates are
        # ``omega_w_eff = omega_w * write`` (per-channel); the
        # per-batch scalar ``omega_w`` and per-channel ``write`` are
        # recovered without division.
        if omega_w.dim() == 3:
            ow_for_dist = omega_w  # [B, T, 1] broadcasts to [B, T, D]
        else:
            ow_for_dist = omega_w.unsqueeze(-1) if omega_w.dim() == 1 else omega_w
        if omega_e.dim() == 3:
            oe_for_dist = omega_e
        else:
            oe_for_dist = omega_e.unsqueeze(-1) if omega_e.dim() == 1 else omega_e
        d_omega_w = (d_ow_eff * write).sum(dim=-1, keepdim=True)
        d_omega_e = (d_oe_eff * erase).sum(dim=-1, keepdim=True)
        d_write = d_ow_eff * ow_for_dist
        d_erase = d_oe_eff * oe_for_dist
        if omega_w.dim() == 2:
            d_omega_w = d_omega_w.squeeze(-1)
        if omega_e.dim() == 2:
            d_omega_e = d_omega_e.squeeze(-1)
        return (
            dstate_in,
            dks,
            dvs,
            d_omega_w,
            d_omega_e,
            d_write,
            d_erase,
            None,
        )


def chunkwise_delta_core_step_out(
    state0: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega_w: torch.Tensor,
    omega_e: torch.Tensor,
    write: torch.Tensor,
    erase: torch.Tensor,
    next_state: torch.Tensor,
) -> torch.Tensor:
    r"""Differentiable chunkwise RewardNet state update.

    Applies the per-step RewardNet update over the time axis and
    saves per-token states for the Taichi adjoint. Equivalent to
    calling :func:`delta_core_step_out` for every token but
    exposes the full per-token state vector to the bwd kernel so the
    chunkwise adjoint can be evaluated in a single Taichi launch.

    Input shapes:

        state0  : ``[B, d, d]``  initial state
        k       : ``[B, T, d]``  per-token keys
        v       : ``[B, T, d]``  per-token values
        omega_w : ``[B, T, 1]``  per-batch scalar plasticity (write)
        omega_e : ``[B, T, 1]``  per-batch scalar plasticity (erase)
        write   : ``[B, T, d]``  per-token, per-dim write gate
        erase   : ``[B, T, d]``  per-token, per-dim erase gate

    Returns the final state ``next_state`` shape ``[B, d, d]``.
    """
    return _ChunkwiseDeltaCoreStepFunction.apply(
        state0,
        k,
        v,
        omega_w,
        omega_e,
        write,
        erase,
        next_state,
    )


__all__ = [
    "chunkwise_delta_core_step_out",
    "delta_core_step_out",
    "launch_chunk_bwd_per_bh",
    "launch_delta_core_step",
    "launch_delta_core_step_bwd",
]
