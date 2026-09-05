"""InfiniDopamineGatedRewardNet: reward-augmented variant of GatedRewardNet.

Moved from ``decoder_layer.py`` for size.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.cache_utils import Cache

from qwendopamine.models.infinidopamine.configs import (
    InfiniDopamineConfig,
    InfiniDopamineTextConfig,
)
from qwendopamine.models.reinforced import (
    GatedRewardNet,
    GatedRewardNetConfig,
)


class InfiniDopamineGatedRewardNet(GatedRewardNet):
    r"""InfiniDopamineGatedRewardNet(config, layer_idx, k_stats=6, **kwargs) -> None

    InfiniDopamine reward-augmented variant of :class:`GatedRewardNet`.

    Args:
        config (InfiniDopamineConfig | InfiniDopamineTextConfig): Layer config.
        layer_idx (int): Layer index for cache disambiguation.
        k_stats (int): Number of reward statistics. Default: ``6``.
        **kwargs: Extra args forwarded to :class:`GatedRewardNet`.
    """

    def __init__(
        self,
        config: InfiniDopamineConfig | InfiniDopamineTextConfig,
        layer_idx: int,
        k_stats: int = 6,
        **kwargs: Any,
    ) -> None:
        reward_net_config = GatedRewardNetConfig(
            hidden_size=config.hidden_size,
            k_stats=k_stats,
            layer_idx=layer_idx,
            conv_size=getattr(config, "linear_conv_kernel_dim", 4),
            norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            reward_dropout=getattr(config, "reward_dropout", 0.0),
            advantage_dropout=getattr(config, "advantage_dropout", 0.0),
            hidden_dropout=getattr(
                config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
            ),
            memory_rank=getattr(config, "reward_memory_rank", None),
            **kwargs,
        )
        super().__init__(reward_net_config)
        self.config = config
        self.key_dim = getattr(config, "linear_key_head_dim", 128) * getattr(
            config, "linear_num_key_heads", 16
        )
        self.value_dim = getattr(config, "linear_value_head_dim", 128) * getattr(
            config, "linear_num_value_heads", 32
        )
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self._register_load_state_dict_pre_hook(self._convert_gdn1_weights_hook)

    def _convert_gdn1_weights_hook(
        self, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        qkvz_key = prefix + "in_proj_qkvz.weight"
        qkv_key = prefix + "in_proj_qkv.weight"
        ba_key = prefix + "in_proj_ba.weight"
        b_key = prefix + "in_proj_b.weight"
        a_key = prefix + "in_proj_a.weight"
        z_key = prefix + "in_proj_z.weight"
        conv_key = prefix + "conv1d.weight"
        dt_key = prefix + "dt_bias"
        alog_key = prefix + "A_log"
        norm_key = prefix + "norm.weight"
        out_key = prefix + "out_proj.weight"

        is_gdn1 = any(
            k in state_dict
            for k in (qkvz_key, qkv_key, ba_key, b_key, a_key, z_key, out_key)
        )
        if not is_gdn1:
            return

        if out_key in state_dict:
            out_w = state_dict.pop(out_key)
            if out_w.shape != self.output_proj.weight.shape:
                if (
                    out_w.shape[0] == self.output_proj.weight.shape[0]
                    and out_w.shape[1] >= self.output_proj.weight.shape[1]
                ):
                    out_w = out_w[:, : self.output_proj.weight.shape[1]]
                else:
                    out_w = self.output_proj.weight.data.clone()
            state_dict[prefix + "output_proj.weight"] = out_w

        if qkvz_key in state_dict:
            qkvz = state_dict.pop(qkvz_key)
            qkv = qkvz[: self.conv_dim]
        elif qkv_key in state_dict:
            qkv = state_dict.pop(qkv_key)
        else:
            qkv = None

        if qkv is not None:
            q_w, k_w, v_w = torch.split(
                qkv, [self.key_dim, self.key_dim, self.value_dim], dim=0
            )
            if q_w.shape[0] == self.hidden_size:
                state_dict[prefix + "delta_layer.q_proj.weight"] = q_w
            if k_w.shape[0] == self.hidden_size:
                state_dict[prefix + "delta_layer.memory_core.k_proj.weight"] = k_w
            if v_w.shape[0] == self.hidden_size:
                state_dict[prefix + "delta_layer.memory_core.v_proj.weight"] = v_w

        for old_k in (
            ba_key,
            b_key,
            a_key,
            z_key,
            conv_key,
            dt_key,
            alog_key,
            norm_key,
        ):
            state_dict.pop(old_k, None)

        for param_name, param in self.named_parameters():
            full_key = prefix + param_name
            if full_key not in state_dict:
                state_dict[full_key] = param.data.clone()

    def _update_reward_cache(
        self,
        cache_params: Cache,
        new_cache: dict[str, Any],
    ) -> None:
        r"""Write the freshly computed reward state back into ``cache_params``.

        Uses reward-specific cache fields so the GDN-2 branch state is never
        clobbered. Falls back to the generic field names when the cache
        layer does not yet expose the dedicated reward fields (e.g. dict
        caches returned by a non-HF caller).
        """
        layer_idx = self.layer_idx
        if layer_idx is None:
            return
        if not hasattr(cache_params, "layers"):
            return
        layer_cache = cache_params.layers[layer_idx]
        layer_cache.reward_recurrent_state = new_cache["recurrent_state"]
        layer_cache.reward_value_baseline = new_cache["value_baseline"]
        layer_cache.reward_conv_states = new_cache["conv_state"]
        # Reward normalisation EMA statistics persist when present.
        running_mean = new_cache.get("running_mean")
        running_std = new_cache.get("running_std")
        if running_mean is not None:
            layer_cache.reward_running_mean = running_mean
        if running_std is not None:
            layer_cache.reward_running_std = running_std

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        reward_values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""forward(hidden_states, cache_params=None, attention_mask=None, reward_values=None, **kwargs) -> torch.Tensor

        Apply GatedRewardNet recurrence and persist reward cache.

        Args:
            hidden_states (torch.Tensor): Input ``[B, T, D]``.
            cache_params (Cache | None): HF cache to read/write states.
            attention_mask (torch.Tensor | None): Padding mask.
            reward_values (torch.Tensor | None): Reward input ``[B, T]``.
            **kwargs: Extra kwargs forwarded to the parent.

        Returns:
            torch.Tensor: ``[B, T, D]`` output.
        """
        use_cache = kwargs.pop("use_cache", cache_params is not None)
        out, _, new_cache = super().forward(
            hidden_states=hidden_states,
            reward_values=reward_values,
            past_key_values=cache_params,
            use_cache=use_cache,
            **kwargs,
        )
        if use_cache and cache_params is not None and new_cache is not None:
            self._update_reward_cache(cache_params, new_cache)
        return out
