r"""External integrations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .huggingface import (
        GDN2HFBlock,
        GDN2HFConfig,
        HFIntegration,
        InfiniDopamineGDN2HFConfig,
        Qwen35GDN2HFConfig,
    )
else:
    try:
        from .huggingface import (
            GDN2HFBlock,
            GDN2HFConfig,
            HFIntegration,
            InfiniDopamineGDN2HFConfig,
            Qwen35GDN2HFConfig,
        )
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        HFIntegration = None
        GDN2HFConfig = None
        GDN2HFBlock = None
        Qwen35GDN2HFConfig = None
        InfiniDopamineGDN2HFConfig = None

try:
    from .safetensors import load_safetensors, save_safetensors
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    save_safetensors = None
    load_safetensors = None

__all__ = [
    "GDN2HFBlock",
    "GDN2HFConfig",
    "HFIntegration",
    "InfiniDopamineGDN2HFConfig",
    "Qwen35GDN2HFConfig",
    "load_safetensors",
    "save_safetensors",
]
