"""Optional hardware-accelerated GDN-2 execution kernels (Triton / FLA).

These modules are imported lazily by ``qwendopamine.models.gdn2.gdn2`` so the
pure-PyTorch fallbacks remain the default on CPU-only environments.
"""
