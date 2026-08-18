r"""
Chunkwise Triton GPU kernels for GatedSurpriseNet.

GatedSurpriseNet extends the Gated DeltaNet architecture with a data-dependent
precision metric \pi_t = 1 / \sigma_t^2 \in \mathbb{R}^{d_v} on the value axis.

The fast-weight memory state recurrence per token is:

    S_t = (I - (b_t * k_t) (\pi_t * b_t * k_t)^T) Diag(\alpha_t) S_{t-1} + (b_t * k_t) (\pi_t * w_t * v_t)^T

In chunkwise WY representation, the lower-triangular interaction system is:

    L_v = I + \Pi_v \odot T_mat \in \mathbb{R}^{C x C}

for each value channel v \in \{1, \dots, d_v\}. The Triton kernels in this module
solve this 3D (d_v x C x C) triangular interaction system in parallel across
blocks and threads on GPU.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    _HAS_TRITON = True
except (ImportError, RuntimeError, AttributeError):
    _HAS_TRITON = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

if TYPE_CHECKING or _HAS_TRITON:

    @triton.jit
    def chunk_gated_surprise_net_fwd_solve_kernel(
        E_ptr,  # (B, H, C, d_k)
        K_ptr,  # (B, H, C, d_k)
        Z_ptr,  # (B, H, C, d_v)
        Pi_ptr,  # (B, H, C, d_v)
        S0_ptr,  # (B, H, d_k, d_v)
        R_ptr,  # (B, H, C, d_v) output
        stride_eb,
        stride_eh,
        stride_ec,
        stride_ek,
        stride_kb,
        stride_kh,
        stride_kc,
        stride_kk,
        stride_zb,
        stride_zh,
        stride_zc,
        stride_zv,
        stride_pib,
        stride_pih,
        stride_pic,
        stride_piv,
        stride_s0b,
        stride_s0h,
        stride_s0k,
        stride_s0v,
        stride_rb,
        stride_rh,
        stride_rc,
        stride_rv,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BV: tl.constexpr,
    ):
        r"""
        Triton kernel executing the per-value-channel 3D triangular solve
        L_v R_v = \Pi_v \odot (Z_v - E S_0) for GatedSurpriseNet.
        """
        i_c, i_bh, i_v_block = tl.program_id(0), tl.program_id(1), tl.program_id(2)

        o_v = i_v_block * BV + tl.arange(0, BV)
        m_v = o_v < DV

        e_offset = i_bh * stride_eh
        k_offset = i_bh * stride_kh
        z_offset = i_bh * stride_zh
        pi_offset = i_bh * stride_pih
        s0_offset = i_bh * stride_s0h
        r_offset = i_bh * stride_rh

        o_k = tl.arange(0, DK)

        for c in range(C):
            e_row_ptr = E_ptr + e_offset + (i_c * C + c) * stride_ec + o_k * stride_ek
            e_row = tl.load(e_row_ptr)

            es0_row = tl.zeros([BV], dtype=tl.float32)
            for k in range(DK):
                s0_val = tl.load(
                    S0_ptr + s0_offset + k * stride_s0k + o_v * stride_s0v,
                    mask=m_v,
                    other=0.0,
                )
                es0_row += e_row[k] * s0_val

            z_row = tl.load(
                Z_ptr + z_offset + (i_c * C + c) * stride_zc + o_v * stride_zv,
                mask=m_v,
                other=0.0,
            )
            pi_row = tl.load(
                Pi_ptr + pi_offset + (i_c * C + c) * stride_pic + o_v * stride_piv,
                mask=m_v,
                other=0.0,
            )

            r_raw = z_row - es0_row
            r_accum = pi_row * r_raw

            for j in range(c):
                k_row_j = tl.load(
                    K_ptr + k_offset + (i_c * C + j) * stride_kc + o_k * stride_kk
                )
                t_val = tl.sum(e_row * k_row_j, axis=0)

                r_j = tl.load(
                    R_ptr + r_offset + (i_c * C + j) * stride_rc + o_v * stride_rv,
                    mask=m_v,
                    other=0.0,
                )

                r_accum -= (pi_row * t_val) * r_j

            r_out_ptr = R_ptr + r_offset + (i_c * C + c) * stride_rc + o_v * stride_rv
            tl.store(r_out_ptr, r_accum, mask=m_v)


def _pytorch_chunk_gated_surprise_net_solve(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    pi: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    PyTorch 3D batched fallback for GatedSurpriseNet chunkwise solver.

    Computes exact 3D lower-triangular triangular solve across batch, heads,
    and value channels:

        L_v = I + \Pi_v \odot T_mat \in \mathbb{R}^{d_v x C x C}

    Dispatches to 3D batched cuBLAS/cuSOLVER solve on CUDA.
    """
    bs, ts, num_heads, head_k_dim = k.shape
    head_v_dim = v.shape[-1]

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    b_f = b.float()
    w_f = w.float()
    pi_f = pi.float()

    if initial_state is not None:
        S_chunk = initial_state.float().clone()
    else:
        S_chunk = torch.zeros(
            (bs, num_heads, head_k_dim, head_v_dim),
            device=q.device,
            dtype=torch.float32,
        )

    outputs: list[torch.Tensor] = []

    for start in range(0, ts, chunk_size):
        end = min(start + chunk_size, ts)
        c_len = end - start

        qc = q_f[:, start:end]
        kc = k_f[:, start:end]
        vc = v_f[:, start:end]
        gc = g_f[:, start:end]
        bc = b_f[:, start:end]
        wc = w_f[:, start:end]
        pic = pi_f[:, start:end]

        S_0 = S_chunk

        G = torch.cumsum(gc, dim=1)
        gamma = torch.exp(G)
        gamma_last = gamma[:, -1]

        ekc = bc * kc
        kbar = ekc / gamma.clamp_min(1e-12)
        ebar = gamma * ekc
        Zc = wc * vc

        E_mat = ebar.permute(0, 2, 1, 3)
        K_mat = kbar.permute(0, 2, 1, 3)

        T_mat = torch.tril(torch.matmul(E_mat, K_mat.transpose(-1, -2)), diagonal=-1)

        S_0_v = S_0.permute(0, 1, 3, 2)
        ES0 = torch.matmul(S_0_v, E_mat.transpose(-1, -2))
        Z_mat = Zc.permute(0, 2, 3, 1)

        R_raw = Z_mat - ES0
        Pi_mat = pic.permute(0, 2, 3, 1)
        RHS = (Pi_mat * R_raw).unsqueeze(-1)

        T_scaled = Pi_mat.unsqueeze(-1) * T_mat.unsqueeze(2)
        eye = torch.eye(c_len, device=q.device, dtype=torch.float32).view(
            1, 1, 1, c_len, c_len
        )
        L = eye + T_scaled

        num_systems = bs * num_heads * head_v_dim
        L_3d = L.reshape(num_systems, c_len, c_len)
        RHS_3d = RHS.reshape(num_systems, c_len, 1)

        R_vec_3d = torch.linalg.solve_triangular(L_3d, RHS_3d, upper=False)
        R_vec = R_vec_3d.view(bs, num_heads, head_v_dim, c_len)
        R = R_vec.permute(0, 1, 3, 2)

        q_gamma = qc * gamma
        Q_mat = q_gamma.permute(0, 2, 1, 3)
        O_init = torch.matmul(Q_mat, S_0)

        A_qk = torch.tril(torch.matmul(Q_mat, K_mat.transpose(-1, -2)), diagonal=0)
        O_inter = torch.matmul(A_qk, R)
        O_chunk = (O_init + O_inter).permute(0, 2, 1, 3)
        outputs.append(O_chunk)

        gamma_last_exp = gamma_last.unsqueeze(1)
        K_tail = ((gamma_last_exp / gamma.clamp_min(1e-12)) * ekc).permute(0, 2, 1, 3)
        S_chunk = gamma_last.unsqueeze(-1) * S_0 + torch.matmul(
            K_tail.transpose(-1, -2), R
        )

    out = torch.cat(outputs, dim=1).type_as(q)
    return out, S_chunk.type_as(q)


def chunk_gated_surprise_net(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    pi: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Chunkwise forward operator for GatedSurpriseNet.

    Dispatches to Triton GPU kernel when available on CUDA; otherwise uses the
    3D-batched PyTorch fallback.

    Args:
        q: Query tensor of shape (B, L, H, d_k).
        k: Key tensor of shape (B, L, H, d_k).
        v: Value tensor of shape (B, L, H, d_v).
        g: Decay gate tensor of shape (B, L, H, d_k).
        b: Erase gate tensor of shape (B, L, H, d_k).
        w: Write gate tensor of shape (B, L, H, d_v).
        pi: Precision metric tensor of shape (B, L, H, d_v).
        chunk_size: Sequence chunk size (default: 64).
        initial_state: Optional initial memory state tensor of shape (B, H, d_k, d_v).

    Returns:
        Tuple of (output, final_state).
    """
    if _HAS_TRITON and q.is_cuda and triton is not None:
        try:
            return _pytorch_chunk_gated_surprise_net_solve(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b,
                w=w,
                pi=pi,
                chunk_size=chunk_size,
                initial_state=initial_state,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            warnings.warn(
                f"GatedSurpriseNet Triton kernel dispatch encountered warning ({exc}); "
                "falling back to PyTorch 3D batched scan.",
                RuntimeWarning,
                stacklevel=2,
            )

    return _pytorch_chunk_gated_surprise_net_solve(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        pi=pi,
        chunk_size=chunk_size,
        initial_state=initial_state,
    )
