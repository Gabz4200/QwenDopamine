"""Independent oracle re-implementation of GDN-2 for cross-validation.

This oracle is a *third* implementation of the GDN-2 per-step update, in
addition to:

  1. :func:`qwendopamine.ops.references.gdn2_reference.gdn2_reference_step`
     (the hand-derived reference, plain ``torch.einsum`` form).
  2. :func:`qwendopamine.ops.gdn2.torch_recurrent_gdn2` (the production
     recurrent path with the standard optimisations).

The oracle is implemented with explicit per-index Python loops over the
H, K, V axes, with no ``einsum`` and no batched matmul. Because the
canonical reference and the production path both use ``einsum``-style
reductions, a hand-loop implementation is unlikely to share any bug
with them. A passing comparison proves the spec is correctly
implemented, not just that two different ``einsum`` formulations
agree.

Shapes match the canonical reference:

    S     : [B, H, K, V]
    q, k  : [B, H, K]
    v, w  : [B, H, V]
    b     : [B, H, K]
    a     : [B, H, K]   (decay = exp(g))

The algorithm (paper Eq. 10):

    S_dec[k, v] = a[k] * S[k, v]              # column-wise decay
    e[k]       = b[k] * k[k]                 # erased key
    v_ret[v]   = sum_k S_dec[k, v] * e[k]     # retrieved value
    v_new[v]   = w[v] * v[v] - v_ret[v]       # corrected new value
    S_next[k, v] = S_dec[k, v] + k[k] * v_new[v]  # rank-1 update
    y_t[v]     = sum_k S_next[k, v] * q[k]    # readout
"""

from __future__ import annotations

import torch


def gdn2_oracle_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    b_t: torch.Tensor,
    w_t: torch.Tensor,
    a_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token GDN-2 forward via explicit Python loops.

    Returns ``(y_t, S_next)``. No ``einsum``, no batched matmul; the
    only ops are scalar ``+`` / ``*``. The numerical contract must
    match the canonical reference within fp32 tolerance.
    """
    B, H, K, V = S.shape
    y_t = torch.empty(B, H, V, dtype=S.dtype, device=S.device)
    S_next = torch.empty(B, H, K, V, dtype=S.dtype, device=S.device)

    for b in range(B):
        for h in range(H):
            # Pre-compute the column-wise decay once per head.
            a_kv = [float(a_t[b, h, k]) for k in range(K)]
            e_k = [float(b_t[b, h, k]) * float(k_t[b, h, k]) for k in range(K)]
            k_k = [float(k_t[b, h, k]) for k in range(K)]
            q_k = [float(q_t[b, h, k]) for k in range(K)]
            w_v = [float(w_t[b, h, v]) for v in range(V)]
            v_v = [float(v_t[b, h, v]) for v in range(V)]
            S_kv = [[float(S[b, h, k, v]) for v in range(V)] for k in range(K)]

            # S_dec[k, v] = a_kv[k] * S[k, v]
            S_dec = [[a_kv[k] * S_kv[k][v] for v in range(V)] for k in range(K)]
            # v_ret[v] = sum_k S_dec[k, v] * e_k[k]
            v_ret = [sum(S_dec[k][v] * e_k[k] for k in range(K)) for v in range(V)]
            # v_new[v] = w[v] * v[v] - v_ret[v]
            v_new = [w_v[v] * v_v[v] - v_ret[v] for v in range(V)]
            # S_next[k, v] = S_dec[k, v] + k[k] * v_new[v]
            S_next_bh = [
                [S_dec[k][v] + k_k[k] * v_new[v] for v in range(V)] for k in range(K)
            ]
            # y_t[v] = sum_k S_next[k, v] * q[k]
            y_bh = [sum(S_next_bh[k][v] * q_k[k] for k in range(K)) for v in range(V)]

            for v in range(V):
                y_t[b, h, v] = y_bh[v]
                for k in range(K):
                    S_next[b, h, k, v] = S_next_bh[k][v]

    return y_t, S_next


__all__ = ["gdn2_oracle_step"]
