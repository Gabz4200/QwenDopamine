"""GDN-2 (Gated DeltaNet-2) mixer and block for QwenDopamine.

Reference:
  Hatamizadeh et al. (2026). "Gated DeltaNet-2: Decoupling Erase and Write in
  Linear Attention." arXiv:2605.22791.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from qwendopamine.models.normalization import RMSNorm

try:
    from flash_linear_attention.ops.gdn2 import chunk_gdn2  # type: ignore[import]
except ImportError:
    chunk_gdn2 = None  # optional dependency; falls back to pure-PyTorch recurrence


@dataclass
class GDN2Projections:
    r"""Precomputed per-head projections for the GDN-2 recurrence.

    The mixer produces all projections once and passes them to the dispatch
    backend so the backend never touches the projection layers directly.
    """

    q: torch.Tensor   # [B, H, T, d_k]
    k: torch.Tensor   # [B, H, T, d_k]
    v: torch.Tensor   # [B, V, T, d_v]
    alpha: torch.Tensor  # [B, H, T, d_k]
    b: torch.Tensor   # [B, H, T, d_k]
    w: torch.Tensor   # [B, V, T, d_v]


def _gated_delta_rule_2_fallback(
    mixer: GDN2Mixer,
    hidden_states: torch.Tensor,
    proj: GDN2Projections,
) -> torch.Tensor:
    r"""Pure-PyTorch token-by-token Gated Delta Rule-2 recurrence (Eq. 9).

    Standalone function matching the kernel-function pattern used by
    ``torch_recurrent_gated_delta_rule`` in Transformers' Qwen3.5/GatedDeltaNet.

    Args:
        mixer: :class:`GDN2Mixer` whose output norm/projection are applied.
        hidden_states: input tensor of shape ``[B, T, hidden_size]``.
        proj: precomputed per-head projections.

    Returns:
        Output tensor of shape ``[B, T, hidden_size]``.
    """
    B, T, _ = hidden_states.shape
    S = hidden_states.new_zeros(B, mixer.num_heads, mixer.head_dim, mixer.head_v_dim)
    outputs: list[torch.Tensor] = []

    for t in range(T):
        q_t = proj.q[:, :, t]   # [B, H, d_k]
        k_t = proj.k[:, :, t]   # [B, H, d_k]
        v_t = proj.v[:, :, t]   # [B, V, d_v]
        b_t = proj.b[:, :, t]   # [B, H, d_k]
        w_t = proj.w[:, :, t]   # [B, V, d_v]
        a_t = proj.alpha[:, :, t]  # [B, H, d_k]

        e_t = b_t * k_t          # [B, H, d_k]
        z_t = w_t * v_t          # [B, V, d_v]

        S = a_t.unsqueeze(-1) * S
        r_t = torch.matmul(S.transpose(-2, -1), e_t.unsqueeze(-1)).squeeze(-1)  # [B, H, d_v]
        S = S + k_t.unsqueeze(-1) * (z_t - r_t).unsqueeze(-2)
        o_t = torch.matmul(S.transpose(-2, -1), q_t.unsqueeze(-1)).squeeze(-1)  # [B, H, d_v]
        outputs.append(o_t)

    out = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().view(B, T, mixer.value_dim)
    g = mixer.g_proj(hidden_states)
    out = mixer.o_norm(out) * F.silu(g)
    return mixer.o_proj(out)


class _ShortConv(nn.Module):
    r"""Depthwise Conv1d with causal left padding and SiLU activation.

    Operates on ``[B, T, C]`` input; internally transposes to ``[B, C, T]``
    for the depthwise convolution.  Padding is applied only on the left (past
    tokens) so the kernel never sees future information.
    """

    def __init__(self, channels: int, kernel_size: int = 4, bias: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            bias=bias,
            groups=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = F.silu(self.conv(x))
        # -> [B, T, C]
        return x.transpose(1, 2)


class GDN2Mixer(nn.Module):
    r"""Token-mixing core of Gated DeltaNet-2 (arXiv:2605.22791).

    Projects hidden states to query, key, value, erase gate ``b``, write
    gate ``w``, and log-decay ``g``.  Maintains a fixed-size recurrent state
    ``S`` and updates it with the Gated Delta Rule-2 recurrence:

    .. math::

        S_h^{(t)} \leftarrow \operatorname{Diag}(a_t^{(h)}) S_h^{(t-1)}
            + k_t^{(h)} \otimes (z_t^{(h)} - r_t^{(h)})

    where :math:`a_t^{(h)} = \exp(-\alpha_t^{(h)})`, :math:`e_t^{(h)} = b_t^{(h)} \odot k_t^{(h)}`,
    :math:`z_t^{(h)} = w_t^{(h)} \odot v_t^{(h)}`,
    and :math:`r_t^{(h)} = (S_h^{(t-1)})^T e_t^{(h)}`.

    Dispatch follows the pattern of ``nn.MultiheadAttention.forward``: a single
    method with inline fastpath selection, calling standalone kernel functions.

    Args:
        config: object with model-level attributes forwarded via ``getattr``.
            Required keys: ``hidden_size``, ``num_heads``, ``head_dim``.
            Optional keys: ``expand_v``, ``num_v_heads``, ``rms_norm_eps``,
            ``allow_neg_eigval``, ``conv_size``, ``conv_bias``,
            ``gdn2_kernel_mode``.
        layer_idx: layer index for debugging / caching hooks.
    """

    def __init__(self, config: object, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size: int = getattr(config, "hidden_size", 256)

        # Head layout
        self.num_heads: int = getattr(config, "num_heads", 4)
        self.head_dim: int = getattr(config, "head_dim", 32)
        self.expand_v: float = getattr(config, "expand_v", 1.0)
        self.num_v_heads: int = getattr(config, "num_v_heads", None) or self.num_heads
        self.head_v_dim: int = int(self.head_dim * self.expand_v)
        self.key_dim: int = self.num_heads * self.head_dim
        self.value_dim: int = self.num_v_heads * self.head_v_dim

        conv_size: int = getattr(config, "conv_size", 4)
        conv_bias: bool = getattr(config, "conv_bias", False)
        eps: float = getattr(config, "rms_norm_eps", 1e-6)

        # Projections + short conv (paper §3.1)
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.q_conv = _ShortConv(self.key_dim, kernel_size=conv_size, bias=conv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_conv = _ShortConv(self.key_dim, kernel_size=conv_size, bias=conv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.v_conv = _ShortConv(self.value_dim, kernel_size=conv_size, bias=conv_bias)

        # Channel-wise gates (Eq. 11)
        self.b_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # Log-decay (Eq. 12)
        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # Gated output norm
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNorm(self.value_dim, eps=eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Options
        self.allow_neg_eigval: bool = getattr(config, "allow_neg_eigval", False)
        mode = getattr(config, "gdn2_kernel_mode", "fallback")
        if mode not in {"fallback", "chunk"}:
            raise ValueError(f"gdn2_kernel_mode must be 'fallback' or 'chunk'; got {mode!r}")
        self.kernel_mode: str = mode

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        r"""Run the GDN-2 token mixer.

        Inline dispatch: if ``kernel_mode == "chunk"`` and the input is on
        CUDA with ``flash-linear-attention`` installed, use the chunkwise
        kernel; otherwise fall back to the pure-PyTorch recurrence.

        Args:
            hidden_states: float tensor of shape ``[B, T, hidden_size]``.
            **kwargs: unused; present for interface compatibility with other
                attention layers.

        Returns:
            Output tensor of shape ``[B, T, hidden_size]``.
        """
        B, T, _ = hidden_states.shape

        q_flat = self._proj_q(hidden_states)
        k_flat = self._proj_k(hidden_states)
        v_flat = self._proj_v(hidden_states)
        b_flat = self._proj_erase_gate(hidden_states)
        w_flat = self._proj_write_gate(hidden_states)
        alpha_flat = self._proj_log_decay(hidden_states)

        # [B, T, D] -> [B, heads, T, head_dim]
        q = q_flat.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_flat.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_flat.view(B, T, self.num_v_heads, self.head_v_dim).transpose(1, 2)
        b = b_flat.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        w = w_flat.view(B, T, self.num_v_heads, self.head_v_dim).transpose(1, 2)
        alpha = alpha_flat.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        proj = GDN2Projections(q, k, v, alpha, b, w)

        if self.kernel_mode == "chunk" and hidden_states.device.type == "cuda" and chunk_gdn2 is not None:
            o, _ = chunk_gdn2(
                q=q, k=k, v=v, g=alpha, b=b, wg=w,
                scale=self.head_dim**-0.5, chunk_size=64, cu_seqlens=None,
            )
            out = o.transpose(1, 2).contiguous().view(B, T, self.value_dim)
            g = self.g_proj(hidden_states)
            out = self.o_norm(out) * F.silu(g)
            return self.o_proj(out)

        return _gated_delta_rule_2_fallback(self, hidden_states, proj)

    # Projection helpers

    def _proj_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_conv(self.q_proj(hidden_states))
        return F.normalize(q, p=2, dim=-1)

    def _proj_k(self, hidden_states: torch.Tensor) -> torch.Tensor:
        k = self.k_conv(self.k_proj(hidden_states))
        return F.normalize(k, p=2, dim=-1)

    def _proj_v(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.v_conv(self.v_proj(hidden_states))

    def _proj_erase_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b = torch.sigmoid(self.b_proj(hidden_states))
        if self.allow_neg_eigval:
            b = b * 2.0
        return b

    def _proj_write_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.w_proj(hidden_states))

    def _proj_log_decay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        f = self.f_proj(hidden_states)  # [B, T, key_dim]
        A_log = self.A_log.repeat_interleave(self.head_dim)  # [key_dim]
        g = -A_log.view(1, 1, -1) * F.softplus(f + self.dt_bias)
        return torch.exp(g)  # alpha, [B, T, key_dim]


class GatedDeltaNet2Block(nn.Module):
    r"""Full GDN-2 residual block: pre-norm -> mixer -> post-norm -> MLP.

    Matches the interface of the other blocks in this repo
    (forward takes hidden_states and returns hidden_states).
    """

    def __init__(self, config: object, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = getattr(config, "hidden_size", 256)
        eps = getattr(config, "rms_norm_eps", 1e-6)

        self.input_layernorm = RMSNorm(self.hidden_size, eps=eps)
        self.mixer = GDN2Mixer(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=eps)
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
