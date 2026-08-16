r"""QwenDopamine research framework."""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]


def main() -> None:
    """CLI dispatcher: delegate to the Hydra-backed training entrypoint."""
    from qwendopamine.cli.train import main as train_main

    train_main()
