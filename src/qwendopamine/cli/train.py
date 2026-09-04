from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from qwendopamine import DEFAULT_QWEN35_REPO
from qwendopamine.integrations.huggingface import HFIntegration


def _get_cfg(config: DictConfig, *keys: str, default: Any = None) -> Any:
    for k in keys:
        val = OmegaConf.select(config, k, default=None)
        if val is not None:
            return val
    return default


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[3] / "configs"),
    config_name="train/cpu",
)
def main(config: DictConfig) -> None:
    r"""main(config: DictConfig) -> None

    Hydra-backed training CLI entrypoint.

    Loads a HuggingFace model (with optional quantization) and prints a
    confirmation message.

    Args:
        config (DictConfig): Hydra configuration. Supports flat keys and
            nested ``train.*`` sections via :func:`_get_cfg`.

    Returns:
        None
    """
    print(OmegaConf.to_yaml(config))
    quantization_config = None
    if _get_cfg(
        config, "quantization.enabled", "train.quantization.enabled", default=False
    ):
        quantization_config = HFIntegration.make_quantization_config(
            method=_get_cfg(
                config,
                "quantization.method",
                "train.quantization.method",
                default="int8",
            )
        )
    base_model = _get_cfg(
        config,
        "model.base_model",
        "train.model.base_model",
        default=DEFAULT_QWEN35_REPO,
    )
    device = _get_cfg(config, "train.device", "model.device", "device", default="cpu")
    model = HFIntegration.load_model(
        base_model,
        quantization_config=quantization_config,
        device_map=device,
    )
    print(
        f"Model {type(model).__name__} loaded with quantization={quantization_config is not None}"
    )


if __name__ == "__main__":
    main()
