from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train/cpu")
def main(config: DictConfig) -> None:
    print(f"Evaluate entrypoint stub with config: {config}")
