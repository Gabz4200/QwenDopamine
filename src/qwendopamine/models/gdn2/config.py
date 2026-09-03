# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GDN2Config:
    r"""Configuration for GDN-2 blocks.

    Args:
        hidden_size: Dimension of the hidden states ``[B, T, H, D]``.
        num_heads: Number of query attention heads.
        head_dim: Dimension per head ``D``.
        num_v_heads: Number of value heads (``None`` defaults to ``num_heads``).
        expand_v: Expansion factor for the value projection ``v``.
        conv_size: Kernel size of the causal ``ShortConvolution`` pre-filter.
        conv_bias: If ``True``, add a bias to the convolution.
        allow_neg_eigval: If ``True``, allow negative eigenvalues in the state matrix.
        norm_eps: Epsilon value for ``RMSNorm`` normalization.
        block_size: Maximum sequence length supported by the chunkwise recurrent path.
        vocab_size: Size of the vocabulary.
        num_layers: Number of ``GatedDeltaNet2`` layers stacked in the model.
    """

    name: str = ""
    hidden_size: int = 2048
    num_heads: int = 16
    head_dim: int = 128
    num_v_heads: int | None = None
    expand_v: float = 1.0
    conv_size: int = 4
    conv_bias: bool = False
    allow_neg_eigval: bool = False
    norm_eps: float = 1e-5
    block_size: int = 4096
    vocab_size: int = 32000
    num_layers: int = 24

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> GDN2Config:
        r"""Instantiate a :class:`GDN2Config` from a registered name.

        Args:
            name: Registered config name (e.g. ``"gdn2_1.3B"``).
            **kwargs: Additional fields to override the default values for the
                named config.

        Returns:
            A :class:`GDN2Config` instance with the requested settings.

        Raises:
            KeyError: If ``name`` is not a registered config.
        """
        if name not in name_to_config:
            raise KeyError(
                f"Unknown config name '{name}'. Available configs: {list(name_to_config.keys())}"
            )
        conf_dict: dict[str, Any] = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)


configs = [
    {
        "name": "gdn2_1.3B",
        "block_size": 4096,
        "vocab_size": 32000,
        "hidden_size": 2304,
        "num_heads": 18,
        "head_dim": 128,
        "num_layers": 24,
        "conv_size": 4,
        "norm_eps": 1e-5,
    },
]

name_to_config = {config["name"]: config for config in configs}
