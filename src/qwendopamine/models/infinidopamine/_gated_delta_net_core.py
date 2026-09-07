"""Core InfiniDopamine GatedDeltaNet class.

Extracted from ``_gated_delta_net.py`` for size and SRP.

The decorator stack ``@use_kernel_forward_from_hub`` /
``@use_kernelized_func`` is preserved from the original.
"""

from __future__ import annotations

from typing import Any, Unpack

import torch
from transformers.cache_utils import Cache
from transformers.integrations import (
    use_kernel_forward_from_hub,
    use_kernelized_func,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextGatedDeltaNet,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from transformers.utils.generic import TransformersKwargs

from qwendopamine.models.gdn2 import torch_chunk_gdn2, torch_recurrent_gdn2
from qwendopamine.models.infinidopamine._gated_delta_net_gating import (
    gate_entropy,
    gate_regularization_loss,
)
from qwendopamine.models.infinidopamine.configs import (
    InfiniDopamineTextConfig,
)


@use_kernel_forward_from_hub("InfiniDopamineGatedDeltaNet")
@use_kernelized_func(
    [
        torch_chunk_gdn2,
        torch_recurrent_gdn2,
        causal_conv1d_fn,
        causal_conv1d_update,
    ]
)
class InfiniDopamineGatedDeltaNet(Qwen3NextGatedDeltaNet):
    r"""InfiniDopamineGatedDeltaNet(config, layer_idx) -> None

    This is a framework adapter. It exists so the rest of the codebase
    can depend on a project-owned abstraction instead of the concrete
    ``transformers`` implementation. If the upstream delta-net API changes,
    only this module needs updating.

    InfiniDopamine linear-attention layer with adaptive gating and
    per-head gate entropy monitoring.

    Args:
        config: InfiniDopamineTextConfig.
        layer_idx: Index of this layer within the model.
    """

    def __init__(
        self,
        config: InfiniDopamineTextConfig,
        layer_idx: int,
    ) -> None:
        super().__init__(config, layer_idx)
        self._register_load_state_dict_pre_hook(self._convert_gdn1_weights_hook)

    def _convert_gdn1_weights_hook(
        self, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        b_key = prefix + "in_proj_b.weight"
        a_key = prefix + "in_proj_a.weight"
        qkv_key = prefix + "in_proj_qkv.weight"
        z_key = prefix + "in_proj_z.weight"
        conv1d_key = prefix + "conv1d.weight"

        if b_key in state_dict:
            state_dict[b_key] = state_dict[b_key].contiguous()
        if a_key in state_dict:
            state_dict[a_key] = state_dict[a_key].contiguous()
        if qkv_key in state_dict:
            state_dict[qkv_key] = state_dict[qkv_key].contiguous()
        if z_key in state_dict:
            state_dict[z_key] = state_dict[z_key].contiguous()
        if conv1d_key in state_dict:
            state_dict[conv1d_key] = state_dict[conv1d_key].contiguous()

    def fix_query_key_value_ordering(self) -> None:
        r"""fix_query_key_value_ordering() -> None

        InfiniDopamineGatedDeltaNet uses fused QKV projections;
        no checkpoint reordering is required.
        """
        raise NotImplementedError(
            "InfiniDopamineGatedDeltaNet uses fused QKV projections; "
            "no checkpoint reordering is required."
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""forward(hidden_states: torch.Tensor, cache_params: Cache | None = None, attention_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor

        Args:
            hidden_states: Input hidden states of shape
                ``(batch, seq_len, hidden_size)``.
            cache_params: Optional cache for KV states.
            attention_mask: Optional attention mask tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            Output hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        result = super().forward(hidden_states, cache_params, attention_mask, **kwargs)
        return result

    def get_gate_regularization_loss(
        self, target: float = 0.5, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Compute mean squared deviation of data-dependent routing gates from target balance.

        Args:
            target: Target gate value (default ``0.5``).
            hidden_states: Optional hidden states to compute gates from.

        Returns:
            Scalar tensor of mean squared deviation.
        """
        return gate_regularization_loss(
            model=self, target=target, hidden_states=hidden_states
        )

    def get_gate_entropy(
        self, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Compute Shannon entropy of the routing gate distribution across heads and tokens.

        Args:
            hidden_states: Optional hidden states to compute gates from.

        Returns:
            Scalar tensor of mean entropy across heads and tokens.
        """
        return gate_entropy(model=self, hidden_states=hidden_states)

    def _get_swa_mask(
        self,
        seq_len: int,
        sliding_window: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        r"""Return a sliding-window causal mask, caching by (seq_len, sliding_window, device, dtype).

        Args:
            seq_len: Sequence length.
            sliding_window: Size of the sliding window.
            device: Optional device for the mask tensor.
            dtype: Optional dtype for the mask tensor.

        Returns:
            A causal mask tensor of shape ``(seq_len, seq_len)``.
        """
        from qwendopamine.models.infinidopamine._gated_delta_net_utils import (
            get_swa_mask as _get_swa_mask,
        )
        return _get_swa_mask(
            seq_len=seq_len,
            sliding_window=sliding_window,
            device=device,
            dtype=dtype,
        )