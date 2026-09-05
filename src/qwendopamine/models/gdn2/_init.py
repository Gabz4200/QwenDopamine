"""Init helpers for GDN-2.

Extracted from :mod:`block` for modularity. These helpers convert a config
object (or explicit args) into the kwargs that
:meth:`GatedDeltaNet2.__init__` consumes, and validate backend selection.
"""

from __future__ import annotations

from typing import Any, Literal

from qwendopamine.models.gdn2._backend_helpers import _GATED_DELTA_NET_BACKENDS


def build_init_kwargs(
    hidden_size_or_config: int | Any = 2048,
    hidden_size: int | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
    layer_idx: int | None = None,
    mode: Literal["chunk", "fused_recurrent"] = "chunk",
    expand_v: float = 1.0,
    num_v_heads: int | None = None,
    use_short_conv: bool = True,
    allow_neg_eigval: bool = False,
    conv_size: int = 4,
    conv_bias: bool = False,
    norm_eps: float = 1e-5,
    chunk_size: int = 64,
    backend: str = "auto",
    compile_backend: bool = False,
    fp32_decay: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    r"""build_init_kwargs(...) -> dict[str, Any]

    Convert the public init signature into the kwargs dict consumed by
    :meth:`GatedDeltaNet2.__init__`. Accepts a config object as the first
    positional argument or explicit field-by-field overrides.

    Args:
        hidden_size_or_config (int | Any): First positional. A config object
            (anything with ``hidden_size`` or ``n_embd``) or an int.
        hidden_size (int | None): Hidden dimension. Default: ``None``.
        num_heads (int | None): Number of query heads. Default: ``None``.
        head_dim (int | None): Per-head dimension. Default: ``None``.
        layer_idx (int | None): Layer index. Default: ``None``.
        mode (Literal): ``"chunk"`` or ``"fused_recurrent"``. Default: ``"chunk"``.
        expand_v (float): Value head expansion. Default: ``1.0``.
        num_v_heads (int | None): Number of value heads. Default: ``None``.
        use_short_conv (bool): Use the short-conv pre-filter. Default: ``True``.
        allow_neg_eigval (bool): Allow negative eigenvalues. Default: ``False``.
        conv_size (int): Conv kernel size. Default: ``4``.
        conv_bias (bool): Conv bias. Default: ``False``.
        norm_eps (float): RMS norm epsilon. Default: ``1e-5``.
        chunk_size (int): Chunk size. Default: ``64``.
        backend (str): Backend identifier. Default: ``"auto"``.
        compile_backend (bool): Use ``torch.compile``. Default: ``False``.
        fp32_decay (bool): Upcast decay to float32. Default: ``True``.
        **kwargs: Extra fields forwarded.

    Returns:
        dict[str, Any]: Resolved init kwargs.

    Raises:
        ValueError: If ``backend`` is not in the known set.
    """
    if hasattr(hidden_size_or_config, "hidden_size") or hasattr(
        hidden_size_or_config, "n_embd"
    ):
        cfg = hidden_size_or_config
        hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 2048))
        num_heads = getattr(cfg, "num_heads", getattr(cfg, "n_head", 16))
        head_dim = getattr(cfg, "head_dim", getattr(cfg, "head_size", 128))
        num_v_heads = getattr(
            cfg,
            "num_v_heads",
            getattr(cfg, "n_query_groups", num_v_heads or num_heads),
        )
        conv_size = getattr(
            cfg, "conv_size", getattr(cfg, "conv_kernel_size", conv_size)
        )
        norm_eps = getattr(cfg, "norm_eps", getattr(cfg, "rms_norm_eps", norm_eps))
        allow_neg_eigval = getattr(cfg, "allow_neg_eigval", allow_neg_eigval)
        expand_v = getattr(cfg, "expand_v", expand_v)
        chunk_size = getattr(
            cfg, "chunk_size", getattr(cfg, "train_chunk_size", chunk_size)
        )
        backend = getattr(cfg, "backend", backend)
        compile_backend = getattr(cfg, "compile_backend", compile_backend)
        fp32_decay = getattr(cfg, "fp32_decay", fp32_decay)
    elif hidden_size is None:
        hidden_size = int(hidden_size_or_config)

    if backend not in _GATED_DELTA_NET_BACKENDS:
        raise ValueError(
            f"Invalid GDN-2 backend '{backend}'. "
            f"Valid backends: {list(_GATED_DELTA_NET_BACKENDS)}"
        )

    return {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_v_heads": num_v_heads,
        "layer_idx": layer_idx,
        "mode": mode,
        "expand_v": expand_v,
        "use_short_conv": use_short_conv,
        "allow_neg_eigval": allow_neg_eigval,
        "conv_size": conv_size,
        "conv_bias": conv_bias,
        "norm_eps": norm_eps,
        "chunk_size": chunk_size,
        "backend": backend,
        "compile_backend": compile_backend,
        "fp32_decay": fp32_decay,
    }


__all__ = ["build_init_kwargs"]
