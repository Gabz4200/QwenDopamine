"""Test that the kernels module is importable and exposes expected symbols."""

from qwendopamine.kernels.taichi import is_available


def test_kernels_module_importable():
    """Verify qwendopamine.kernels.taichi is importable and exposes expected symbols."""
    from qwendopamine.kernels.taichi import (
        chunk_taichi_gdn2,
        delta_core_step_out,
        recurrent_taichi_gdn2,
        require,
        taichi_arch,
    )
    from qwendopamine.kernels.taichi import (
        is_available as _is_avail,
    )

    # Verify all symbols are accessible and have the expected types
    assert callable(is_available), "is_available should be callable"
    assert callable(taichi_arch), "taichi_arch should be callable"
    assert callable(require), "require should be callable"
    assert callable(chunk_taichi_gdn2), "chunk_taichi_gdn2 should be callable"
    assert callable(recurrent_taichi_gdn2), "recurrent_taichi_gdn2 should be callable"
    assert callable(delta_core_step_out), (
        "delta_core_step_out should be callable"
    )

    # Verify is_available returns a bool
    assert isinstance(_is_avail(), bool), "is_available() should return bool"

    # Verify taichi_arch returns a string
    arch = taichi_arch()
    assert isinstance(arch, str), f"taichi_arch() should return str, got {type(arch)}"
