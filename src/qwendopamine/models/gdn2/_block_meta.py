# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.
"""GDN-2 block metadata constants.

Extracted from :mod:`qwendopamine.models.gdn2.block` for SRP:
this module owns default hyper-parameters, while
:class:`qwendopamine.models.gdn2.block.GatedDeltaNet2` owns the layer logic.
The Taichi probe stays in ``block.py`` for H4 lazy-import compliance.
"""

from __future__ import annotations

_DEFAULT_HIDDEN_SIZE = 2048
_DEFAULT_NUM_HEADS = 16
_DEFAULT_HEAD_DIM = 128
_DEFAULT_MODE = "chunk"
_DEFAULT_EXPAND_V = 1.0
_DEFAULT_USE_SHORT_CONV = True
_DEFAULT_ALLOW_NEG_EIGVAL = False
_DEFAULT_CONV_SIZE = 4
_DEFAULT_CONV_BIAS = False
_DEFAULT_NORM_EPS = 1e-5
_DEFAULT_CHUNK_SIZE = 64
_DEFAULT_BACKEND = "auto"
_DEFAULT_COMPILE_BACKEND = False
_DEFAULT_FP32_DECAY = True

__all__ = [
    "_DEFAULT_ALLOW_NEG_EIGVAL",
    "_DEFAULT_BACKEND",
    "_DEFAULT_CHUNK_SIZE",
    "_DEFAULT_COMPILE_BACKEND",
    "_DEFAULT_CONV_BIAS",
    "_DEFAULT_CONV_SIZE",
    "_DEFAULT_EXPAND_V",
    "_DEFAULT_FP32_DECAY",
    "_DEFAULT_HEAD_DIM",
    "_DEFAULT_HIDDEN_SIZE",
    "_DEFAULT_MODE",
    "_DEFAULT_NORM_EPS",
    "_DEFAULT_NUM_HEADS",
    "_DEFAULT_USE_SHORT_CONV",
]
