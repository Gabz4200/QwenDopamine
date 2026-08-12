from __future__ import annotations

import os
from typing import Any

import torch
from gguf import GGUFReader, dequantize

from .huggingface import HFIntegration


GGUF_TO_HF_NAME_MAP: dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
}

_ATTN_TENSOR_MAP: dict[str, str] = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_b.weight",
    "ssm_beta.weight": "linear_attn.in_proj_a.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "nextn.hnorm.weight": "nextn.hnorm.weight",
    "nextn.enorm.weight": "nextn.enorm.weight",
    "nextn.shared_head_norm.weight": "nextn.shared_head_norm.weight",
    "nextn.eh_proj.weight": "nextn.eh_proj.weight",
}


def _map_gguf_name_to_hf(gguf_name: str) -> str | None:
    hf_name = GGUF_TO_HF_NAME_MAP.get(gguf_name)
    if hf_name is not None:
        return hf_name

    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".")
        if len(parts) >= 3:
            layer_idx = parts[1]
            tensor_name = ".".join(parts[2:])
            suffix = _ATTN_TENSOR_MAP.get(tensor_name)
            if suffix is not None:
                return f"model.layers.{layer_idx}.{suffix}"

    return None


def _dequantize_gguf_tensor(tensor: Any, hf_name: str) -> torch.Tensor:
    dequantized = dequantize(tensor.data, tensor.tensor_type)
    if hf_name.endswith(".conv1d.weight") and dequantized.ndim == 2:
        return torch.from_numpy(dequantized.copy()).unsqueeze(1)
    return torch.from_numpy(dequantized.copy())


def _build_state_dict_from_gguf(gguf_path: str) -> dict[str, torch.Tensor]:
    reader = GGUFReader(gguf_path)
    state_dict: dict[str, torch.Tensor] = {}

    for tensor in reader.tensors:
        hf_name = _map_gguf_name_to_hf(tensor.name)
        if hf_name is None:
            raise KeyError(f"No HF mapping for GGUF tensor: {tensor.name}")

        state_dict[hf_name] = _dequantize_gguf_tensor(tensor, hf_name)

    return state_dict


def load_gguf_weights(model: Any, gguf_path: str) -> None:
    state_dict = _build_state_dict_from_gguf(gguf_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        allowed_missing = {"lm_head.weight"}
        unexpected_missing = set(missing) - allowed_missing
        if unexpected_missing:
            raise RuntimeError(f"Missing keys after GGUF load: {sorted(unexpected_missing)}")


def convert_gguf_to_safetensors(gguf_path: str, output_dir: str) -> str:
    from safetensors.torch import save_file

    state_dict = _build_state_dict_from_gguf(gguf_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/model.safetensors"
    save_file(state_dict, output_path)
    return output_path


def load_qwen35_from_gguf(
    model_name: str = "Qwen/Qwen3.5-0.8B",
    config: Any = None,
    device_map: str = "cpu",
    **kwargs: Any,
) -> Any:
    """Load Qwen3.5 from standard HF weights by default.

    Args:
        model_name: HF repo ID or path to .gguf file.
        config: Optional pre-built config. Defaults to Qwen/Qwen3.5-0.8B.
        device_map: Device placement string.
        **kwargs: Extra kwargs forwarded to HFIntegration.load_model.

    Returns:
        HF model with weights loaded.
    """
    if not model_name.endswith(".gguf"):
        return HFIntegration.load_model(
            model_name=model_name,
            quantization_config=None,
            device_map=device_map,
            **kwargs,
        )

    gguf_path = model_name
    if config is None:
        config = HFIntegration.load_config("Qwen/Qwen3.5-0.8B")
    model = HFIntegration.load_model(
        "Qwen/Qwen3.5-0.8B",
        quantization_config=None,
        device_map=device_map,
        **kwargs,
    )
    load_gguf_weights(model, gguf_path)
    return model
