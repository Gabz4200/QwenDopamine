# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GDN2Config:
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
