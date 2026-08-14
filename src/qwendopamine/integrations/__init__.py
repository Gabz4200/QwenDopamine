r"""External integrations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .huggingface import (
        GatedSurpriseNetHFBlock,
        GatedSurpriseNetHFConfig,
        GDN2HFBlock,
        GDN2HFConfig,
        HFIntegration,
    )
else:
    try:
        from .huggingface import (
            GatedSurpriseNetHFBlock,
            GatedSurpriseNetHFConfig,
            GDN2HFBlock,
            GDN2HFConfig,
            HFIntegration,
        )
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        HFIntegration = None
        GDN2HFConfig = None
        GDN2HFBlock = None
        GatedSurpriseNetHFConfig = None
        GatedSurpriseNetHFBlock = None

try:
    from .safetensors import load_safetensors, save_safetensors
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    save_safetensors = None
    load_safetensors = None

__all__ = [
    "GDN2HFBlock",
    "GDN2HFConfig",
    "GatedSurpriseNetHFBlock",
    "GatedSurpriseNetHFConfig",
    "HFIntegration",
    "load_safetensors",
    "save_safetensors",
]
