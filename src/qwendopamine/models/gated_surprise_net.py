from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from qwendopamine.models.gdn2.gdn2 import RMSNormGated, ShortConvolution

if TYPE_CHECKING:
    from transformers.cache_utils import Cache
else:
    try:
        from transformers.cache_utils import Cache
    except ImportError:
        Cache = Any

try:
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
except ImportError:
    LinearAttentionCacheLayerMixin = type(None)  # type: ignore[misc, assignment]


@dataclass
class SurpriseRecurrenceState:
    memory: torch.Tensor
    first_moment: torch.Tensor
    second_moment: torch.Tensor
    step: torch.Tensor


def l2_normalize_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Normalize tensor along the last dimension using L2 norm."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def gaussian_nll_diag(
    target: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
    full: bool = False,
) -> torch.Tensor:
    r"""Compute diagonal Gaussian Negative Log-Likelihood."""
    var = var.clamp_min(eps)
    loss = 0.5 * (torch.log(var) + (target - mean).square() / var)
    if full:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    return loss.sum(dim=-1)


def torch_get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Pure PyTorch helper for unpadding 2D attention masks."""
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() > 0 else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def torch_index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    r"""Index first axis of a tensor using flat indices."""
    return x[indices]


def torch_pad_input(
    hidden_states: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int
) -> torch.Tensor:
    r"""Pad flattened sequence back to [batch_size, seq_len, ...] shape."""
    output_shape = (batch_size * seq_len,) + hidden_states.shape[1:]
    padded = torch.zeros(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
    padded[indices] = hidden_states
    return padded.view(batch_size, seq_len, *hidden_states.shape[1:])


class SurpriseMemoryAdam(nn.Module):
    r"""Local Adam surprise recurrence memory update on diagonal Gaussian NLL."""

    def __init__(
        self,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        nll_var_eps: float = 1e-6,
        nll_full: bool = False,
        learnable_init: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.nll_var_eps = nll_var_eps
        self.nll_full = nll_full

        init = torch.zeros(num_heads, head_k_dim, head_v_dim)
        if learnable_init:
            self.memory_init = nn.Parameter(init)
        else:
            self.register_buffer("memory_init", init)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SurpriseRecurrenceState:
        device = device if device is not None else self.memory_init.device
        dtype = dtype if dtype is not None else torch.float32
        mem0 = self.memory_init.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        m0 = torch.zeros_like(mem0)
        v0 = torch.zeros_like(mem0)
        t0 = torch.zeros(batch_size, self.num_heads, 1, 1, device=device, dtype=dtype)
        return SurpriseRecurrenceState(mem0, m0, v0, t0)

    def one_step(
        self,
        state: SurpriseRecurrenceState,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        g_t: torch.Tensor,
        b_t: torch.Tensor,
        w_t: torch.Tensor,
    ) -> tuple[SurpriseRecurrenceState, torch.Tensor, torch.Tensor]:
        out_dtype = q_t.dtype
        q_f = q_t.float()
        k_f = k_t.float()
        v_f = v_t.float()
        g_f = g_t.float()
        b_f = b_t.float()
        w_f = w_t.float()

        alpha_t = torch.exp(g_f)
        S_bar = alpha_t.unsqueeze(-1) * state.memory

        erase_key = b_f * k_f
        target_mu = w_f * v_f

        pred_mu = torch.einsum("bhkv,bhk->bhv", S_bar, erase_key)
        pred_logvar = torch.log1p(pred_mu.square())
        pred_var = F.softplus(pred_logvar) + self.nll_var_eps

        residual_scaled = (pred_mu - target_mu) / pred_var
        grad = torch.einsum("bhk,bhv->bhkv", erase_key, residual_scaled)

        t = state.step + 1.0
        m = self.beta1 * state.first_moment + (1.0 - self.beta1) * grad
        v = self.beta2 * state.second_moment + (1.0 - self.beta2) * grad.square()

        bias_c1 = 1.0 - torch.pow(torch.full_like(t, self.beta1), t)
        bias_c2 = 1.0 - torch.pow(torch.full_like(t, self.beta2), t)
        m_hat = m / bias_c1.clamp_min(1e-12)
        v_hat = v / bias_c2.clamp_min(1e-12)

        memory_new = S_bar - self.lr * m_hat / (v_hat.sqrt() + self.eps)
        out_t = torch.einsum("bhkv,bhk->bhv", memory_new, q_f).to(out_dtype)
        nll_t = gaussian_nll_diag(
            target=target_mu,
            mean=pred_mu,
            var=pred_var,
            eps=self.nll_var_eps,
            full=self.nll_full,
        ).to(out_dtype)
        new_state = SurpriseRecurrenceState(memory_new, m, v, t)
        return new_state, out_t, nll_t

    def serial_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        initial_state: SurpriseRecurrenceState | None = None,
        detach_state_every_step: bool = False,
    ) -> tuple[torch.Tensor, SurpriseRecurrenceState, torch.Tensor]:
        bs = k.shape[0]
        if initial_state is None:
            state = self.initial_state(bs, device=k.device, dtype=torch.float32)
        else:
            state = SurpriseRecurrenceState(
                initial_state.memory.float(),
                initial_state.first_moment.float(),
                initial_state.second_moment.float(),
                initial_state.step.float(),
            )
        outputs: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for i in range(k.shape[1]):
            state, out_i, nll_i = self.one_step(state, q[:, i], k[:, i], v[:, i], g[:, i], b[:, i], w[:, i])
            outputs.append(out_i)
            losses.append(nll_i)
            if detach_state_every_step:
                state = SurpriseRecurrenceState(
                    state.memory.detach(),
                    state.first_moment.detach(),
                    state.second_moment.detach(),
                    state.step.detach(),
                )
        return torch.stack(outputs, dim=1), state, torch.stack(losses, dim=1)

    def chunk_parallel_training_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        chunk_size: int,
        initial_state: SurpriseRecurrenceState | None = None,
    ) -> tuple[torch.Tensor, SurpriseRecurrenceState, torch.Tensor]:
        bs, ts = k.shape[:2]
        state = (
            self.initial_state(bs, device=k.device, dtype=torch.float32)
            if initial_state is None
            else initial_state
        )
        outputs: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for start in range(0, ts, chunk_size):
            end = min(start + chunk_size, ts)
            out_chunk, state, nll_chunk = self.serial_scan(
                q=q[:, start:end],
                k=k[:, start:end],
                v=v[:, start:end],
                g=g[:, start:end],
                b=b[:, start:end],
                w=w[:, start:end],
                initial_state=state,
                detach_state_every_step=False,
            )
            outputs.append(out_chunk)
            losses.append(nll_chunk)
        return torch.cat(outputs, dim=1), state, torch.cat(losses, dim=1)


class GatedSurpriseNetAdam(nn.Module):
    r"""GatedSurpriseNet drop-in token mixer using local Adam surprise recurrence.

    This module maintains constructor signature and forward contract compatibility
    with QwenDopamine transformer blocks while running device-agnostically on CPU and GPU.
    """

    def __init__(
        self,
        hidden_size_or_config: int | Any = 2048,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        layer_idx: int | None = None,
        mode: Literal["chunk", "fused_recurrent"] = "chunk",
        expand_v: float = 1.0,
        num_v_heads: int | None = None,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        local_adam_lr: float = 1e-3,
        local_adam_beta1: float = 0.9,
        local_adam_beta2: float = 0.999,
        local_adam_eps: float = 1e-8,
        nll_var_eps: float = 1e-6,
        nll_full: bool = False,
        learnable_init_state: bool = False,
        train_chunk_size: int = 128,
        ttt_preserve_serial_recurrence: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if hasattr(hidden_size_or_config, "hidden_size") or hasattr(
            hidden_size_or_config, "n_embd"
        ):
            cfg = hidden_size_or_config
            hidden_size = getattr(
                cfg, "hidden_size", getattr(cfg, "n_embd", 2048)
            )
            num_heads = getattr(
                cfg, "num_heads", getattr(cfg, "n_head", 16)
            )
            head_dim = getattr(
                cfg, "head_dim", getattr(cfg, "head_size", 128)
            )
            num_v_heads = getattr(
                cfg, "num_v_heads", getattr(cfg, "n_query_groups", num_v_heads or num_heads)
            )
            conv_size = getattr(
                cfg, "conv_size", getattr(cfg, "conv_kernel_size", conv_size)
            )
            norm_eps = getattr(
                cfg, "norm_eps", getattr(cfg, "rms_norm_eps", norm_eps)
            )
            allow_neg_eigval = getattr(
                cfg, "allow_neg_eigval", allow_neg_eigval
            )
            expand_v = getattr(cfg, "expand_v", expand_v)
            local_adam_lr = getattr(cfg, "local_adam_lr", local_adam_lr)
            local_adam_beta1 = getattr(cfg, "local_adam_beta1", local_adam_beta1)
            local_adam_beta2 = getattr(cfg, "local_adam_beta2", local_adam_beta2)
            local_adam_eps = getattr(cfg, "local_adam_eps", local_adam_eps)
            nll_var_eps = getattr(cfg, "nll_var_eps", nll_var_eps)
            nll_full = getattr(cfg, "nll_full", nll_full)
            learnable_init_state = getattr(cfg, "learnable_init_state", learnable_init_state)
            train_chunk_size = getattr(cfg, "train_chunk_size", train_chunk_size)
        elif isinstance(hidden_size_or_config, dict):
            cfg_dict = hidden_size_or_config
            hidden_size = cfg_dict.get("hidden_size", 2048)
            num_heads = cfg_dict.get("num_heads", 16)
            head_dim = cfg_dict.get("head_dim", 128)
            num_v_heads = cfg_dict.get("num_v_heads", num_heads)
            conv_size = cfg_dict.get("conv_size", conv_size)
            norm_eps = cfg_dict.get("norm_eps", norm_eps)
            allow_neg_eigval = cfg_dict.get("allow_neg_eigval", allow_neg_eigval)
            expand_v = cfg_dict.get("expand_v", expand_v)
            local_adam_lr = cfg_dict.get("local_adam_lr", local_adam_lr)
            local_adam_beta1 = cfg_dict.get("local_adam_beta1", local_adam_beta1)
            local_adam_beta2 = cfg_dict.get("local_adam_beta2", local_adam_beta2)
            local_adam_eps = cfg_dict.get("local_adam_eps", local_adam_eps)
            nll_var_eps = cfg_dict.get("nll_var_eps", nll_var_eps)
            nll_full = cfg_dict.get("nll_full", nll_full)
            learnable_init_state = cfg_dict.get("learnable_init_state", learnable_init_state)
            train_chunk_size = cfg_dict.get("train_chunk_size", train_chunk_size)
        elif hidden_size is None:
            hidden_size = int(hidden_size_or_config)

        self.hidden_size = hidden_size
        self.num_heads = num_heads if num_heads is not None else 16
        self.head_dim = head_dim if head_dim is not None else 128
        self.num_v_heads = num_v_heads if num_v_heads is not None else self.num_heads
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.layer_idx = layer_idx
        self.norm_eps = norm_eps

        self.local_adam_lr = local_adam_lr
        self.local_adam_beta1 = local_adam_beta1
        self.local_adam_beta2 = local_adam_beta2
        self.local_adam_eps = local_adam_eps
        self.nll_var_eps = nll_var_eps
        self.nll_full = nll_full
        self.learnable_init_state = learnable_init_state
        self.train_chunk_size = train_chunk_size
        self.ttt_preserve_serial_recurrence = ttt_preserve_serial_recurrence

        self.mode = mode
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)

        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )
        if not math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={self.head_dim}."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        cast(Any, self.A_log)._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        cast(Any, self.dt_bias)._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNormGated(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.memory = SurpriseMemoryAdam(
            num_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            lr=self.local_adam_lr,
            beta1=self.local_adam_beta1,
            beta2=self.local_adam_beta2,
            eps=self.local_adam_eps,
            nll_var_eps=self.nll_var_eps,
            nll_full=self.nll_full,
            learnable_init=self.learnable_init_state,
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        cast(Any, module)._is_hf_initialized = True

    def _get_cache(
        self, past_key_values: Cache | dict[str, Any] | None
    ) -> tuple[
        SurpriseRecurrenceState | None,
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
    ]:
        if past_key_values is None:
            return None, None

        if isinstance(past_key_values, Cache):
            layers = getattr(past_key_values, "layers", [])
            if self.layer_idx is not None and self.layer_idx < len(layers):
                layer_cache = layers[self.layer_idx]
                rec_states = getattr(layer_cache, "recurrent_states", None)
                rec_state = (
                    rec_states[0]
                    if rec_states is not None and len(rec_states) > 0
                    else None
                )
                conv_state = getattr(layer_cache, "conv_states", None)
                return rec_state, conv_state
            return None, None

        if isinstance(past_key_values, dict):
            return past_key_values.get("recurrent_state"), past_key_values.get("conv_state")

        return None, None

    def _update_cache(
        self,
        past_key_values: Cache | dict[str, Any] | None,
        recurrent_state: SurpriseRecurrenceState | None,
        conv_state: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
    ) -> None:
        if past_key_values is None:
            return

        if self.layer_idx is not None and isinstance(past_key_values, Cache):
            layers = getattr(past_key_values, "layers", [])
            if self.layer_idx < len(layers):
                layer_cache = layers[self.layer_idx]
                is_recurrent_layer = (
                    isinstance(layer_cache, LinearAttentionCacheLayerMixin)
                    or hasattr(layer_cache, "update_recurrent_state")
                    or hasattr(layer_cache, "recurrent_states")
                )
                if (
                    is_recurrent_layer
                    and hasattr(past_key_values, "update_recurrent_state")
                    and recurrent_state is not None
                ):
                    past_key_values.update_recurrent_state(cast(Any, recurrent_state), self.layer_idx)
                if (
                    is_recurrent_layer
                    and hasattr(past_key_values, "update_conv_state")
                    and conv_state is not None
                ):
                    past_key_values.update_conv_state(cast(Any, conv_state), self.layer_idx)
        elif isinstance(past_key_values, dict):
            if recurrent_state is not None:
                past_key_values["recurrent_state"] = recurrent_state
            if conv_state is not None:
                past_key_values["conv_state"] = conv_state

    def _project_inputs(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        use_cache: bool = False,
        conv_states: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
    ]:
        new_conv_q, new_conv_k, new_conv_v = None, None, None
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = (
                conv_states if conv_states is not None else (None, None, None)
            )
            q, new_conv_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, new_conv_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, new_conv_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = (
            -self.A_log.float().exp().repeat_interleave(self.head_k_dim)
            * F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        ).to(hidden_states.dtype)
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b_gate = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w_gate = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)

        q = l2_normalize_last(q)
        k = l2_normalize_last(k)

        if self.num_v_heads > self.num_heads:
            groups = self.num_v_heads // self.num_heads
            q, k, g, b_gate = (
                repeat(x, "... h d -> ... (h g) d", g=groups)
                for x in (q, k, g, b_gate)
            )

        if self.allow_neg_eigval:
            b_gate = b_gate * 2.0

        new_conv_states = (new_conv_q, new_conv_k, new_conv_v) if self.use_short_conv else None
        return q, k, v, g, b_gate, w_gate, new_conv_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | dict[str, Any] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | dict[str, Any] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None

        if attention_mask is not None:
            indices, cu_seqlens, _ = torch_get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = torch_index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        recurrent_state, conv_states = self._get_cache(past_key_values)
        should_use_cache = bool(use_cache or past_key_values is not None)

        q, k, v, g, b_gate, w_gate, new_conv_states = self._project_inputs(
            hidden_states,
            cu_seqlens=cu_seqlens,
            use_cache=should_use_cache,
            conv_states=conv_states,
        )

        if self.training:
            mode = "chunk"
        else:
            mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode

        if mode == "chunk":
            out, final_recurrent_state, _ = self.memory.chunk_parallel_training_scan(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b_gate,
                w=w_gate,
                chunk_size=self.train_chunk_size,
                initial_state=recurrent_state,
            )
        elif mode == "fused_recurrent":
            out, final_recurrent_state, _ = self.memory.serial_scan(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b_gate,
                w=w_gate,
                initial_state=recurrent_state,
                detach_state_every_step=False,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if should_use_cache:
            self._update_cache(
                past_key_values,
                recurrent_state=final_recurrent_state,
                conv_state=new_conv_states,
            )

        gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        out = self.o_norm(out, gate)
        out = rearrange(out, "b t h d -> b t (h d)")
        out = self.o_proj(out)

        if attention_mask is not None and indices is not None:
            out = torch_pad_input(out.squeeze(0), indices, batch_size, q_len)

        return out, None, past_key_values
