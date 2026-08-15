"""Behavioral tests for safetensors integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from qwendopamine.integrations.safetensors import load_safetensors, save_safetensors


def test_when_safetensors_saved_and_loaded_then_preserves_tensors() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "sub" / "model.safetensors"
        state_dict = {
            "weight": torch.randn(10, 10),
            "bias": torch.randn(10),
        }

        save_safetensors(state_dict, save_path)
        assert save_path.exists()

        loaded = load_safetensors(save_path, device="cpu")

        assert "weight" in loaded
        assert "bias" in loaded
        assert torch.allclose(loaded["weight"], state_dict["weight"])
        assert torch.allclose(loaded["bias"], state_dict["bias"])
