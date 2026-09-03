r"""QwenDopamine research framework."""

from __future__ import annotations

__version__ = "0.2.0"

DEFAULT_QWEN35_REPO: str = "Qwen/Qwen3.5-0.8B"

__all__ = ["DEFAULT_QWEN35_REPO", "__version__"]


def main() -> None:
    r"""main() -> None

    CLI dispatcher: delegate to the Hydra-backed training entrypoint.

    Imports :func:`qwendopamine.cli.train.main` lazily to avoid pulling the
    full Hydra/transformers stack into ``import qwendopamine``.

    Returns:
        None
    """
    from qwendopamine.cli.train import main as train_main

    train_main()
