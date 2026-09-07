"""Layer-type selection helpers for InfiniDopamineDecoderLayer."""

from __future__ import annotations

from typing import Any, Literal

from qwendopamine.models.infinidopamine.configs import InfiniDopamineTextConfig

_LINEAR_BLOCK_TYPES: frozenset[str] = frozenset({"linear_attention", "gdn2", "gdn"})
_ATTENTION_BLOCK_TYPES: frozenset[str] = frozenset(
    {"full_attention", "sliding_attention"}
)
_REWARD_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "gated_reward_net",
        "reinforced_delta",
        "reward_net",
        "reward_linear_attention",
    }
)


class LayerTypeError(ValueError):
    """Raised when an unsupported block type is requested."""


def resolve_block_type(
    config: InfiniDopamineTextConfig,
    layer_idx: int,
) -> Literal["linear", "attention", "reward"]:
    """Resolve which mixer family this layer should use.

    Returns:
        One of ``"linear"``, ``"attention"``, or ``"reward"``.

    Raises:
        LayerTypeError: If ``config.layer_types[layer_idx]`` is not a
            recognized block type.
    """
    layer_types: Any = config.layer_types
    block_type = layer_types[layer_idx]
    if block_type in _LINEAR_BLOCK_TYPES:
        return "linear"
    if block_type in _ATTENTION_BLOCK_TYPES:
        return "attention"
    if block_type in _REWARD_BLOCK_TYPES:
        return "reward"
    raise LayerTypeError(
        f"Unsupported InfiniDopamine block_type '{block_type}' at "
        f"layer_idx={layer_idx}. Expected one of "
        f"{sorted(_LINEAR_BLOCK_TYPES | _ATTENTION_BLOCK_TYPES | _REWARD_BLOCK_TYPES)}."
    )


def should_enable_parallel_reward(
    config: InfiniDopamineTextConfig,
    layer_idx: int,
) -> bool:
    """Whether the parallel reward branch is enabled for this layer.

    Resolution order:

    1. ``config.parallel_reward_layers`` is the explicit allow-list.
    2. ``config.use_parallel_reward`` opts in to the implicit rule of
       attaching the branch to attention-only layers
       (``full_attention`` / ``sliding_attention``).
    """
    explicit_layers = tuple(getattr(config, "parallel_reward_layers", ()) or ())
    if explicit_layers:
        return layer_idx in explicit_layers
    if not getattr(config, "use_parallel_reward", False):
        return False
    layer_types: Any = config.layer_types
    return layer_types[layer_idx] in _ATTENTION_BLOCK_TYPES


def should_use_sparse_moe(
    config: InfiniDopamineTextConfig,
    layer_idx: int,
) -> bool:
    """Whether this layer should use the sparse MoE MLP."""
    num_experts = getattr(config, "num_experts", None) or 0
    if num_experts <= 0:
        return False
    if layer_idx in getattr(config, "mlp_only_layers", []):
        return False
    return (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0
