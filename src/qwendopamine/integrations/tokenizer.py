from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


def load_qwen35_tokenizer(model_name: str, **kwargs: Any) -> AutoTokenizer:
    r"""Load a Qwen3.5 tokenizer with a fallback repo candidate.

    If ``model_name`` is a GGUF path, the function strips the filename and
    retries against the base repo. If that fails, it falls back to
    ``Qwen/Qwen3.5-0.8B``.

    Args:
        model_name (str): Hugging Face repo id or local GGUF path.
        **kwargs: extra keyword arguments forwarded to
            :meth:`transformers.AutoTokenizer.from_pretrained`.

    Returns:
        AutoTokenizer: loaded tokenizer.

    Raises:
        RuntimeError: if all candidate repos fail to load.
    """
    if model_name.endswith(".gguf"):
        model_name = model_name.rsplit("/", 1)[0]

    repo_candidates = [model_name, "Qwen/Qwen3.5-0.8B"]

    last_error: Exception | None = None
    for candidate in repo_candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, **kwargs)  # type: ignore[return-value]
        except (OSError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(
        "Failed to load Qwen3.5 tokenizer from candidate repos. "
        f"Last error: {last_error}"
    )
