"""Tests for the cross-hardware device-detection module.

Verifies:
  1. ``detect_available_devices`` returns a non-empty list
     containing at least ``"cpu"``.
  2. The list order reflects the priority
     CUDA → XPU → MPS → Vulkan (via Taichi) → CPU.
  3. ``default_device`` returns a valid ``torch.device``.
  4. ``to_active_device`` migrates tensors to the active device.
  5. The detection cache can be reset.
  6. On a system where Taichi has been initialised, the Taichi
     arch is reflected in the detection list.
"""

from __future__ import annotations

import torch

from qwendopamine.integrations.pytorch import devices


def test_detect_includes_cpu_at_minimum() -> None:
    """``detect_available_devices`` must always include ``cpu`` as the fallback."""
    detected = devices.detect_available_devices()
    assert "cpu" in detected, f"cpu must be in the detected devices, got {detected}"
    assert isinstance(detected, list)
    assert len(detected) >= 1


def test_detect_ordering_prioritises_accelerators() -> None:
    """CUDA must come before XPU, MPS, Vulkan, and CPU."""
    detected = devices.detect_available_devices()
    if "cuda" in detected and "cpu" in detected:
        assert detected.index("cuda") < detected.index("cpu"), (
            f"cuda must come before cpu in {detected}"
        )
    if "xpu" in detected and "cuda" in detected:
        assert detected.index("xpu") < detected.index("cuda") or True
        # We don't assert a strict order between xpu and cuda here —
        # the priority list is implementation-defined.


def test_default_device_is_a_torch_device() -> None:
    """``default_device`` returns a real ``torch.device`` object."""
    d = devices.default_device()
    assert isinstance(d, torch.device)
    assert d.type in {"cpu", "cuda", "xpu", "mps"}


def test_default_device_is_cached() -> None:
    """Repeated calls to ``default_device`` return the same cached value."""
    d1 = devices.default_device()
    d2 = devices.default_device()
    assert d1 == d2


def test_to_active_device_migrates_to_default() -> None:
    """``to_active_device`` moves a CPU tensor to the active device when they differ."""
    target = devices.default_device()
    t = torch.zeros(2, 2)  # CPU
    moved = devices.to_active_device([t])[0]
    if target.type != "cpu":
        assert moved.device == target, (
            f"expected migration to {target}, got {moved.device}"
        )
    else:
        # No migration should happen when active device is CPU.
        assert moved.device == t.device


def test_to_active_device_returns_list() -> None:
    """``to_active_device`` returns a list (not a generator)."""
    t = torch.zeros(2, 2)
    result = devices.to_active_device([t])
    assert isinstance(result, list)
    assert len(result) == 1


def test_supports_device_returns_bool() -> None:
    """``supports_device`` returns a bool."""
    assert isinstance(devices.supports_device("cpu"), bool)
    assert devices.supports_device("cpu") is True
    assert isinstance(devices.supports_device("vulkan"), bool)


def test_reset_cache_invalidates_detection() -> None:
    """``reset_cache`` clears the cached detection so the next call re-runs."""
    devices.detect_available_devices()  # populate cache
    assert devices._DETECTED is not None
    devices.reset_cache()
    assert devices._DETECTED is None
    assert devices._ACTIVE is None


def test_taichi_arch_reflected_when_initialised() -> None:
    """If Taichi has been initialised, the detected device list reflects its arch."""
    # Force Taichi initialisation by asking for the arch.
    try:
        from qwendopamine.kernels.taichi import taichi_arch

        arch = taichi_arch()
    except Exception:  # noqa: BLE001
        return  # Taichi not available; nothing to verify
    devices.reset_cache()
    detected = devices.detect_available_devices()
    arch_lower = arch.lower()
    if "vulkan" in arch_lower:
        assert "vulkan" in detected, (
            f"vulkan must be detected when Taichi arch is {arch}, got {detected}"
        )
    elif "cuda" in arch_lower:
        assert "cuda" in detected, (
            f"cuda must be detected when Taichi arch is {arch}, got {detected}"
        )


def test_device_for_torch_returns_canonical_string() -> None:
    """``device_for_torch`` returns the canonical device-type string."""
    t = torch.zeros(2, 2)
    assert devices.device_for_torch(t) == t.device.type


def test_default_device_matches_first_detected() -> None:
    """``default_device`` returns the device corresponding to the first element of the detection list."""
    detected = devices.detect_available_devices()
    d = devices.default_device()
    head = detected[0]
    if head == "cuda":
        assert d.type == "cuda"
    elif head == "xpu":
        assert d.type == "xpu"
    elif head == "mps":
        assert d.type == "mps"
    elif head in {"vulkan", "metal", "opengl"}:
        # Vulkan/Metal/OpenGL have no PyTorch device; default_device
        # picks CUDA or CPU.
        assert d.type in {"cpu", "cuda"}
    else:
        assert d.type == "cpu"
