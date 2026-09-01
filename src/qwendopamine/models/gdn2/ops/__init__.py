"""GDN-2 primitive ops.

Small per-position operations that wrap the recurrence kernel:

- ``conv`` — depthwise causal short convolution with decoding-state cache
- ``norm`` — gated RMS normalization (with and without dtype promotion)
"""

from qwendopamine.models.gdn2.ops.conv import ShortConvolution
from qwendopamine.models.gdn2.ops.norm import RMSNormGated, RMSNormGatedNoCast

__all__ = ["RMSNormGated", "RMSNormGatedNoCast", "ShortConvolution"]