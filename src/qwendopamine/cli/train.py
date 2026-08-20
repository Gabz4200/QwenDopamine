from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from qwendopamine.integrations.huggingface import HFIntegration


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[3] / "configs"),
    config_name="train/cpu",
)
def main(config: DictConfig) -> None:
    r"""Training CLI entrypoint.

    Loads a Hugging Face model with optional quantization and prints a
    confirmation message.

    The accessor paths below assume a ``train/*`` primary config: Hydra groups
    the composed result under the ``train`` key (e.g. ``train.train.device``,
    ``train.quantization.enabled``, ``train.model.base_model``). All lookups
    fall back to safe defaults so configs that omit optional sections (such as
    ``train/single_gpu``, which has no ``quantization`` block) still work.
    """
    print(OmegaConf.to_yaml(config))
    quantization_config = None
    if OmegaConf.select(config, "train.quantization.enabled", default=False):
        quantization_config = HFIntegration.make_quantization_config(
            method=OmegaConf.select(config, "train.quantization.method", default="int8")
        )
    base_model = OmegaConf.select(config, "train.model.base_model", default=None)
    device = OmegaConf.select(config, "train.train.device", default="cpu")
    model = HFIntegration.load_model(
        base_model or "Qwen/Qwen3.5-4B",
        quantization_config=quantization_config,
        device_map=device,
    )
    print(
        f"Model {type(model).__name__} loaded with quantization={quantization_config is not None}"
    )


if __name__ == "__main__":
    main()
