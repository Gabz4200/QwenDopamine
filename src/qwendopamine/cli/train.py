from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from qwendopamine.integrations.huggingface import HFIntegration


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train/cpu")
def main(config: DictConfig) -> None:
    r"""Training CLI entrypoint.

    Loads a Hugging Face model with optional quantization and prints a
    confirmation message.
    """
    print(OmegaConf.to_yaml(config))
    quantization_config = None
    if config.quantization.enabled:
        quantization_config = HFIntegration.make_quantization_config(method=config.quantization.method)
    model = HFIntegration.load_model(
        config.model.base_model or "Qwen/Qwen3.5-4B",
        quantization_config=quantization_config,
        device_map=config.train.device or "cpu",
    )
    print(f"Model loaded with quantization={quantization_config is not None}")


if __name__ == "__main__":
    main()
