r"""External integrations."""

try:
    from .huggingface import GDN2HFBlock, GDN2HFConfig, HFIntegration
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    HFIntegration = None
    GDN2HFConfig = None
    GDN2HFBlock = None

try:
    from .safetensors import load_safetensors, save_safetensors
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    save_safetensors = None
    load_safetensors = None

__all__ = [
    "GDN2HFBlock",
    "GDN2HFConfig",
    "HFIntegration",
    "load_safetensors",
    "save_safetensors",
]
