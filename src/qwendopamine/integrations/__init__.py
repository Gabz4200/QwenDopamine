r"""External integrations."""

try:
    from .huggingface import HFIntegration
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    HFIntegration = None

try:
    from .safetensors import load_safetensors, save_safetensors
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    save_safetensors = None
    load_safetensors = None

__all__ = ["HFIntegration", "load_safetensors", "save_safetensors"]
