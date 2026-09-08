# Copyright (c) 2026, NVIDIA CORPORATION & QwenDopamine Authors.
# Licensed under the Apache License 2.0 or MIT license.

"""Analytical parameter counting for GDN2GPT."""

from __future__ import annotations

from qwendopamine.models.gdn2_gpt.config import GDN2GPTConfig


def compute_model_params(cfg: GDN2GPTConfig) -> dict[str, int]:
    r"""compute_model_params(cfg: GDN2GPTConfig) -> dict[str, int]

    Analytically compute model parameter counts across components.

    Args:
        cfg (GDN2GPTConfig): Model configuration.

    Returns:
        dict[str, int]: Per-component param counts keyed by ``"total"``,
        ``"embed"``, ``"lm_head"``, ``"standard_block"``, ``"gdn2_block"``,
        ``"num_standard_layers"``, ``"num_gdn2_layers"``.
    """
    padded_vocab = cfg.padded_vocab_size or cfg.vocab_size
    embed_params = padded_vocab * cfg.n_embd
    lm_head_params = cfg.n_embd * padded_vocab
    final_norm_params = cfg.n_embd

    mlp_params_per_layer = (cfg.n_embd * cfg.intermediate_size * 2) + (
        cfg.intermediate_size * cfg.n_embd
    )
    norm_params_per_layer = cfg.n_embd * 2

    qkv_dim = (cfg.n_head + 2 * cfg.n_query_groups) * cfg.head_size
    attn_params_per_layer = (cfg.n_embd * qkv_dim) + (cfg.n_embd * cfg.n_embd)

    k_dim = cfg.n_head * cfg.head_size
    hv = int(cfg.head_size * cfg.expand_v)
    v_dim = int(cfg.n_head * hv)
    gdn2_projs = (cfg.n_embd * k_dim * 2) + (cfg.n_embd * v_dim)
    conv_params = (
        (k_dim * cfg.conv_size * 2) + (v_dim * cfg.conv_size)
        if cfg.use_short_conv
        else 0
    )
    f_proj = (cfg.n_embd * hv) + (hv * k_dim)
    b_proj = cfg.n_embd * k_dim
    w_proj = cfg.n_embd * v_dim
    g_proj = (cfg.n_embd * hv) + (hv * v_dim) + v_dim
    o_proj = v_dim * cfg.n_embd
    o_norm = hv
    dt_and_a = cfg.n_head + k_dim
    gdn2_attn_params = (
        gdn2_projs
        + conv_params
        + f_proj
        + b_proj
        + w_proj
        + g_proj
        + o_proj
        + o_norm
        + dt_and_a
    )

    num_gdn2_layers = (
        len(cfg.gdn2_layers)
        if cfg.gdn2_layers is not None
        else (cfg.n_layer // cfg.gdn2_per_layer if cfg.gdn2_per_layer > 0 else 1)
    )
    num_standard_attn_layers = cfg.n_layer - num_gdn2_layers

    total = (
        embed_params
        + lm_head_params
        + final_norm_params
        + (mlp_params_per_layer + norm_params_per_layer) * cfg.n_layer
        + attn_params_per_layer * num_standard_attn_layers
        + gdn2_attn_params * num_gdn2_layers
    )
    return {
        "total": total,
        "embed": embed_params,
        "lm_head": lm_head_params,
        "standard_block": mlp_params_per_layer
        + norm_params_per_layer
        + attn_params_per_layer,
        "gdn2_block": mlp_params_per_layer + norm_params_per_layer + gdn2_attn_params,
        "num_standard_layers": num_standard_attn_layers,
        "num_gdn2_layers": num_gdn2_layers,
    }


__all__ = ["compute_model_params"]
