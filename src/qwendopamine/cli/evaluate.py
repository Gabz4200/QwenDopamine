from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train/cpu")
def main(config: DictConfig) -> None:
    r"""Evaluation CLI entrypoint.

    Entrypoint for model evaluation workflows.
    """
    print(f"Evaluate entrypoint with config: {config}")


if __name__ == "__main__":
    main()
