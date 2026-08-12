from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


def load_qwen35_tokenizer(model_name: str, **kwargs: Any) -> AutoTokenizer:
    if model_name.endswith(".gguf"):
        model_name = model_name.rsplit("/", 1)[0]

    repo_candidates = [model_name, "Qwen/Qwen3.5-0.8B"]

    last_error: Exception | None = None
    for candidate in repo_candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, **kwargs)
        except (OSError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(
        "Failed to load Qwen3.5 tokenizer from candidate repos. "
        f"Last error: {last_error}"
    )
