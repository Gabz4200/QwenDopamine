"""GDN-2 (Gated DeltaNet-2) mixer and block for QwenDopamine.

Reference:
  Hatamizadeh et al. (2026). "Gated DeltaNet-2: Decoupling Erase and Write in
  Linear Attention." arXiv:2605.22791.
  Code: https://github.com/NVlabs/GatedDeltaNet-2

Implements the Gated Delta Rule-2 recurrence from Eq. (9) of the paper:

    S_t = (I - k_t (b_t ⊙ k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t ⊙ v_t)^T
    o_t = S_t^T q_t

where b_t is the channel-wise erase gate and w_t is the channel-wise write
gate.  This implementation uses a pure-PyTorch token-by-token recurrence so
it runs on CPU without Triton kernels.  A chunkwise parallel path can be
added later for training speed.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from qwendopamine.models.normalization import RMSNorm


class _ShortConv(nn.Module):
    """Depthwise Conv1d with causal padding and SiLU activation.

    Matches the paper's "short convolution" on q/k/v pathways.
    """

    def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size,
            bias=bias, groups=hidden_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden_size] -> [B, hidden_size, T]
        x = x.transpose(1, 2)
        # Causal pad on the left (past tokens only).
        pad = self.kernel_size - 1
        x = F.pad(x, (pad, 0))
        x = self.conv(x)
        x = F.silu(x)
        # Back to [B, T, hidden_size].
        return x.transpose(1, 2)


class GDN2Mixer(nn.Module):
    """Token-mixing core of Gated DeltaNet-2.

    Projects the hidden state to query, key, value, erase gate ``b``, write
    gate ``w``, and log-decay ``g``.  Maintains a fixed-size recurrent state
    ``S`` and updates it with the Gated Delta Rule-2 recurrence.

    Args:
        config: any object with ``hidden_size``, ``num_heads``, ``head_dim``,
            ``expand_v``, ``num_v_heads``, ``rms_norm_eps``,
            ``allow_neg_eigval``, ``conv_size``, and ``conv_bias`` attributes
            (or reasonable defaults via ``getattr``).
        layer_idx: layer index for debugging / caching hooks.
    """

    def __init__(self, config: object, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = getattr(config, "hidden_size", 256)

        # Multi-head layout.
        self.num_heads = getattr(config, "num_heads", 4)
        self.head_dim = getattr(config, "head_dim", 32)
        self.expand_v = getattr(config, "expand_v", 1.0)
        self.num_v_heads = getattr(config, "num_v_heads", None) or self.num_heads
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        conv_size = getattr(config, "conv_size", 4)
        conv_bias = getattr(config, "conv_bias", False)

        # Query / key / value projections + short convolutions (paper §3.1).
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.q_conv = _ShortConv(self.key_dim, kernel_size=conv_size, bias=conv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_conv = _ShortConv(self.key_dim, kernel_size=conv_size, bias=conv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.v_conv = _ShortConv(self.value_dim, kernel_size=conv_size, bias=conv_bias)

        # GDN-2 channel-wise gates (Eq. 11 in the paper).
        # b_proj -> erase gate on the key axis [0, 1]^{d_k}.
        # w_proj -> write gate on the value axis [0, 1]^{d_v}.
        self.b_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # Log-decay projection (Eq. 12 in the paper).
        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        # A_log: per-head log-rate.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        # dt_bias: per-channel bias for the softplus step-size.
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Gated output path: g_proj -> RMSNormSwishGate -> o_proj.
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNorm(self.value_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Options.
        self.allow_neg_eigval = getattr(config, "allow_neg_eigval", False)

        # Light init: small uniform for all linears.
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """Small-uniform init for all linear layers (matches reference)."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Run the GDN-2 token mixer.

        Args:
            hidden_states: float tensor of shape ``[B, T, hidden_size]``.
            **kwargs: unused; present for interface compatibility with other
                attention layers.

        Returns:
            Output tensor of shape ``[B, T, hidden_size]``.
        """
        B, T, _ = hidden_states.shape

        # -- projections + short conv + L2-norm (paper §3.1) ---------------
        q = self.q_proj(hidden_states)
        q = self.q_conv(q)
        q = F.normalize(q, p=2, dim=-1)  # L2-norm on query pathway

        k = self.k_proj(hidden_states)
        k = self.k_conv(k)
        k = F.normalize(k, p=2, dim=-1)  # L2-norm on key pathway

        v = self.v_proj(hidden_states)
        v = self.v_conv(v)

        # Reshape to per-head: [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_v_heads, self.head_v_dim).transpose(1, 2)

        b = torch.sigmoid(self.b_proj(hidden_states))  # [B, T, key_dim]
        if self.allow_neg_eigval:
            b = b * 2.0
        b = b.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d_k]

        w = torch.sigmoid(self.w_proj(hidden_states))  # [B, T, value_dim]
        w = w.view(B, T, self.num_v_heads, self.head_v_dim).transpose(1, 2)  # [B, V, T, d_v]

        # Log-decay.
        f = self.f_proj(hidden_states)  # [B, T, key_dim]
        A_log = self.A_log.repeat_interleave(self.head_dim)  # [key_dim]
        g = -A_log.view(1, 1, -1) * F.softplus(f + self.dt_bias)
        alpha = torch.exp(g)  # [B, T, key_dim]
        alpha = alpha.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d_k]

        # -- per-head token-by-token recurrence (Eq. 9) --------------------
        # S: [B, num_heads, head_k_dim, head_v_dim]
        S = hidden_states.new_zeros(B, self.num_heads, self.head_dim, self.head_v_dim)
        outputs: list[torch.Tensor] = []

        for t in range(T):
            q_t = q[:, :, t]   # [B, H, d_k]
            k_t = k[:, :, t]   # [B, H, d_k]
            v_t = v[:, :, t]   # [B, V, d_v]
            b_t = b[:, :, t]   # [B, H, d_k]
            w_t = w[:, :, t]   # [B, V, d_v]
            a_t = alpha[:, :, t]  # [B, H, d_k]

            e_t = b_t * k_t    # [B, H, d_k]
            z_t = w_t * v_t    # [B, V, d_v]

            # Decay: S_h ← diag(a_t[h]) S_h
            S = a_t.unsqueeze(-1) * S  # [B, H, d_k, d_v]

            # Read: r_h = S_h^T @ e_h  (matmul handles expand_v automatically)
            # S: [B, H, d_k, d_v], e_t: [B, H, d_k]
            r_t = torch.matmul(S.transpose(-2, -1), e_t.unsqueeze(-1)).squeeze(-1)
            # r_t: [B, H, d_v]
            # Delta update: S_h ← S_h + k_h ⊗ (z_h - r_h)
            delta = (z_t - r_t).unsqueeze(-2)  # [B, H, 1, d_v]
            S = S + k_t.unsqueeze(-1) * delta
            # Output: o_h = S_h^T @ q_h
            o_t = torch.matmul(S.transpose(-2, -1), q_t.unsqueeze(-1)).squeeze(-1)
            # o_t: [B, H, d_v]

            outputs.append(o_t)

        # outputs: list of [B, num_v_heads, head_v_dim]
        out = torch.stack(outputs, dim=2)  # [B, V, T, d_v]
        out = out.transpose(1, 2).contiguous().view(B, T, self.value_dim)

        # -- gated output norm + projection (paper architecture diagram) -----
        g = self.g_proj(hidden_states)  # [B, T, value_dim]
        out = self.o_norm(out) * F.silu(g)
        out = self.o_proj(out)          # [B, T, hidden_size]
        return out


class GatedDeltaNet2Block(nn.Module):
    """Full GDN-2 residual block: pre-norm -> mixer -> post-norm -> MLP.

    Matches the interface of the other blocks in this repo
    (forward takes hidden_states and returns hidden_states).
    """

    def __init__(self, config: object, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = getattr(config, "hidden_size", 256)
        self.norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.input_layernorm = RMSNorm(self.hidden_size, eps=self.norm_eps)
        self.mixer = GDN2Mixer(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.norm_eps)
        self.mlp = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
