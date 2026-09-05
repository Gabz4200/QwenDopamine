"""Block: hybrid GDN-2 / standard-attention transformer block.

Moved from ``model.py`` for size.
"""

from __future__ import annotations

import torch
from torch import nn

from qwendopamine.models.core.normalization import RMSNorm
from qwendopamine.models.gdn2 import GatedDeltaNet2
from qwendopamine.models.gdn2_gpt.attention import CausalSelfAttention
from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig
from qwendopamine.models.gdn2_gpt.mlp import LLaMAMLP
from qwendopamine.models.gdn2_gpt.rope import KVCache, RoPECache


class Block(nn.Module):
    r"""Transformer block choosing between GDN-2 and standard attention.

    Args:
        config (GDN2GPTConfig): Model configuration.
        layer_idx (int): Index of this layer within the model.
    """

    def __init__(self, config: GDN2GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)

        if config.gdn2_layers is not None:
            self.use_gdn2 = layer_idx in config.gdn2_layers
        elif config.gdn2_per_layer > 0:
            self.use_gdn2 = layer_idx % config.gdn2_per_layer == 0
        else:
            self.use_gdn2 = layer_idx == (config.n_layer // 2)

        if self.use_gdn2:
            self.attn = GatedDeltaNet2(
                hidden_size=config.n_embd,
                num_heads=config.n_head,
                head_dim=config.head_size,
                num_v_heads=config.n_head,
                layer_idx=layer_idx,
                norm_eps=config.norm_eps,
                conv_size=config.conv_size,
                expand_v=config.expand_v,
                use_short_conv=config.use_short_conv,
                chunk_size=config.chunk_size or config.train_chunk_size,
                allow_neg_eigval=config.allow_neg_eigval,
                backend=config.backend,
                fp32_decay=config.fp32_decay,
                compile_backend=config.compile_backend,
            )
        else:
            self.attn = CausalSelfAttention(
                config, layer_idx=layer_idx, n_embd=config.n_embd
            )

        if (
            not config.shared_attention_norm
            and config.mlp
            and not config.parallel_residual
        ):
            self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        if config.mlp:
            self.mlp = LLaMAMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache | None,
        max_seq_length: int,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        r"""forward(x: torch.Tensor, rope: RoPECache | None, max_seq_length: int, mask: torch.Tensor | None = None, input_pos: torch.Tensor | None = None, kv_cache: KVCache | None = None) -> tuple[torch.Tensor, KVCache | None]

        Apply attention (GDN-2 or standard) and MLP sublayer with residuals.

        Args:
            x (torch.Tensor): Input ``[B, T, D]``.
            rope (RoPECache | None): ``(cos, sin)`` cache or ``None``.
            max_seq_length (int): Cache window length.
            mask (torch.Tensor | None): Attention mask.
            input_pos (torch.Tensor | None): Positions for cache indexing.
            kv_cache (KVCache | None): ``(k_cache, v_cache)``.

        Returns:
            tuple[torch.Tensor, KVCache | None]: ``(output, kv_cache)``.
        """
        n_1 = self.norm_1(x)
        if self.use_gdn2:
            h, _, new_kv_cache = self.attn(n_1, attention_mask=None)
        else:
            h, new_kv_cache = self.attn(
                n_1, rope, max_seq_length, mask, input_pos, kv_cache
            )

        if self.config.parallel_residual:
            if self.config.mlp:
                h = h + self.mlp(n_1)
            x = x + h
        else:
            x = x + h
            if self.config.mlp:
                n_2 = self.norm_2(x)
                h = self.mlp(n_2)
                x = x + h
        return x, new_kv_cache
