r"""QwenDopamine research framework."""

from __future__ import annotations

__version__ = "0.2.0"

DEFAULT_QWEN35_REPO: str = "Qwen/Qwen3.5-0.8B"

__all__ = ["__version__", "DEFAULT_QWEN35_REPO"]


def main() -> None:
    """CLI dispatcher: delegate to the Hydra-backed training entrypoint."""
    from qwendopamine.cli.train import main as train_main

    train_main()
