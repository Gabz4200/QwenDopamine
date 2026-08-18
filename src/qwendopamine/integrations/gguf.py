r"""GGUF weight loading and conversion utilities."""

from __future__ import annotations

import os
import re
from typing import Any

import torch

try:
    from gguf import GGUFReader, dequantize
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GGUFReader = None
    dequantize = None

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
    r"""Map a GGUF tensor name to the corresponding HF state-dict key.

    Args:
        gguf_name (str): raw GGUF tensor name.

    Returns:
        str | None: mapped HF key, or ``None`` if unmapped.
    """
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
    r"""Dequantize a GGUF tensor and reshape conv1d weights if needed.

    Args:
        tensor (Any): GGUF tensor metadata.
        hf_name (str): target HF tensor name used to detect conv1d reshape.

    Returns:
        Tensor: dequantized tensor.
    """
    if dequantize is None:
        data = torch.as_tensor(tensor.data)
        if hf_name.endswith(".conv1d.weight") and data.ndim == 2:
            return data.unsqueeze(1)
        return data
    dequantized = dequantize(tensor.data, tensor.tensor_type)
    if hf_name.endswith(".conv1d.weight") and dequantized.ndim == 2:
        return torch.from_numpy(dequantized.copy()).unsqueeze(1)
    return torch.from_numpy(dequantized.copy())


def _build_state_dict_from_gguf(gguf_path: str) -> dict[str, torch.Tensor]:
    r"""Build an HF-style state dict from a GGUF file.

    Args:
        gguf_path (str): path to GGUF file.

    Returns:
        dict[str, Tensor]: mapped state dict.
    """
    if GGUFReader is None or dequantize is None:
        raise RuntimeError(
            "gguf is required for GGUF loading. Install the optional GGUF dependency."
        )
    reader = GGUFReader(gguf_path)
    state_dict: dict[str, torch.Tensor] = {}

    for tensor in reader.tensors:
        hf_name = _map_gguf_name_to_hf(tensor.name)
        if hf_name is None:
            raise KeyError(f"No HF mapping for GGUF tensor: {tensor.name}")

        state_dict[hf_name] = _dequantize_gguf_tensor(tensor, hf_name)

    return state_dict


def load_gguf_weights(model: Any, gguf_path: str) -> None:
    r"""Load GGUF weights into an existing model.

    Missing ``lm_head.weight`` is allowed; all other missing keys raise.

    Args:
        model (Any): model to populate.
        gguf_path (str): path to a GGUF file.
    """
    state_dict = _build_state_dict_from_gguf(gguf_path)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        allowed_missing = {"lm_head.weight"}
        unexpected_missing = set(missing) - allowed_missing
        if unexpected_missing:
            raise RuntimeError(
                f"Missing keys after GGUF load: {sorted(unexpected_missing)}"
            )


def convert_gguf_to_safetensors(gguf_path: str, output_dir: str) -> str:
    r"""Convert a GGUF file to a safetensors file.

    Args:
        gguf_path (str): path to input GGUF file.
        output_dir (str): directory to write ``model.safetensors`` into.

    Returns:
        str: path to written safetensors file.
    """
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
    r"""Load Qwen3.5 from standard HF weights by default, or from GGUF.

    Non-standard GGUF repos (for example ``unsloth/Qwen3.5-0.8B-MTP-GGUF``)
    often lack ``model_type`` in ``config.json``, so they cannot be loaded
    directly with :meth:`transformers.AutoConfig.from_pretrained`. This loader
    builds a base HF model matching the size indicated in ``model_name`` and
    then overlays GGUF weights on top of it.

    Args:
        model_name (str): HF repo ID for the base model, a GGUF repo ID, or a
            path to a ``.gguf`` file. The repo name must contain a size token
            such as ``0.8B``, ``1.7B``, ``4B``, etc., so the base model can
            be inferred.
        config (Any, optional): optional pre-built config. Defaults to the
            config for the inferred base model.
        device_map (str): device placement string. Default: ``"cpu"``.
        **kwargs: extra keyword arguments forwarded to
            :meth:`HFIntegration.load_model`.

    Returns:
        Any: HF model with weights loaded.
    """
    size_match = re.search(
        r"Qwen3\.5-(\d+(?:\.\d+)?)B", model_name, flags=re.IGNORECASE
    )
    if size_match is None:
        raise ValueError(
            f"Could not infer Qwen3.5 size from model_name: {model_name!r}"
        )
    size_token = size_match.group(1)
    base_model = f"Qwen/Qwen3.5-{size_token}B"
    if config is None:
        config = HFIntegration.load_config(base_model)

    model = HFIntegration.load_model(
        base_model,
        quantization_config=None,
        device_map=device_map,
        **kwargs,
    )

    if model_name == base_model or model_name == base_model.replace("/", "-"):
        return model

    if model_name.endswith(".gguf"):
        load_gguf_weights(model, model_name)
    else:
        try:
            from huggingface_hub import list_repo_files

            candidates = [
                f
                for f in list_repo_files(model_name)
                if f.startswith(f"Qwen3.5-{size_token}B-") and f.endswith(".gguf")
            ]
            if not candidates:
                raise RuntimeError(f"No GGUF files found in repo '{model_name}'")
            gguf_file = candidates[0]
            from huggingface_hub import hf_hub_download

            gguf_path = hf_hub_download(model_name, filename=gguf_file)
            load_gguf_weights(model, gguf_path)
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as exc:
            raise RuntimeError(
                f"Cannot load '{model_name}' as a regular HF model and failed to load as GGUF repo: {exc}"
            ) from exc

    return model
