r"""Monitoring helpers for the parallel GatedRewardNet branch.

The training loop and external trainers (e.g. the CPT notebook's
``CPTSFTTrainer``) call :func:`collect_parallel_reward_metrics` to surface
diagnostics about the parallel reward branch without coupling them to the
specific decoder layer implementation. Metrics include:

* ``parallel_reward/gate_mean`` and ``parallel_reward/gate_max`` — sigmoid
  output of the data-dependent gate.
* ``parallel_reward/branch_norm`` — ``||sigmoid(gate) * reward_out||`` (the
  effective contribution to the residual stream).
* ``parallel_reward/main_norm`` — ``||main_out||``.
* ``parallel_reward/branch_ratio`` — ``branch_norm / (main_norm + eps)``.
  Should stay below the ``warn_branch_ratio_above`` threshold early in
  training.
* ``parallel_reward/value_baseline`` — sum of the EMA baseline norm, exposed
  to confirm the cache is being written.
* ``parallel_reward/recurrent_state_norm`` — sum of the fast-weight state
  norm (low-rank aware).
* ``parallel_reward/active_layers`` — number of decoder layers that have a
  parallel reward branch attached.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = [
    "collect_parallel_reward_metrics",
    "maybe_warn_branch_ratio",
]


def _iter_decoder_layers(model: nn.Module) -> list[nn.Module]:
    r"""Locate the decoder layer list regardless of which wrapper owns it."""
    candidate_names = ("layers", "language_model", "model")
    for name in candidate_names:
        holder = getattr(model, name, None)
        if holder is None:
            continue
        if isinstance(holder, nn.ModuleList):
            return list(holder)
        inner = getattr(holder, "layers", None)
        if isinstance(inner, nn.ModuleList):
            return list(inner)
    return []


def _has_parallel_branch(layer: nn.Module) -> bool:
    return hasattr(layer, "reward_branch") and hasattr(layer, "reward_gate_proj")


def _norm(t: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None) -> float:
    if t is None:
        return 0.0
    if isinstance(t, tuple):
        return float(
            sum(x.detach().float().abs().sum().item() for x in t if x is not None)
        )
    if not isinstance(t, torch.Tensor):
        return 0.0
    return float(t.detach().float().abs().sum().item())


def collect_parallel_reward_metrics(
    model: nn.Module,
    main_out: torch.Tensor | None = None,
    reward_out: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    past_key_values: Any = None,
) -> dict[str, float]:
    r"""Aggregate parallel reward branch statistics into a flat dict.

    Args:
        model: The full model. The function locates the decoder layer list
            and inspects each layer for ``reward_branch`` /
            ``reward_gate_proj`` attributes.
        main_out: Optional tensor — the main mixer output for the current
            step. When provided, the branch-to-main ratio uses this tensor.
        reward_out: Optional tensor — the parallel branch output for the
            current step. Combined with ``gate`` to compute the actual
            contribution to the residual stream.
        gate: Optional tensor — the gate values applied to ``reward_out``.
        past_key_values: Optional HF ``DynamicCache``. When provided the
            function reports the live value baseline + recurrent state norm
            of the first layer that has a parallel branch attached, so the
            caller can confirm state persistence.

    Returns:
        Flat mapping of metric name to float. Empty dict when the model has
        no parallel reward branch.
    """
    layers = _iter_decoder_layers(model)
    active_layers = [layer for layer in layers if _has_parallel_branch(layer)]
    if not active_layers:
        return {"parallel_reward/active_layers": 0.0}

    metrics: dict[str, float] = {
        "parallel_reward/active_layers": float(len(active_layers)),
    }

    if main_out is not None and isinstance(main_out, torch.Tensor):
        metrics["parallel_reward/main_norm"] = float(
            main_out.detach().float().abs().sum().item()
        )
    if reward_out is not None and isinstance(reward_out, torch.Tensor):
        metrics["parallel_reward/branch_raw_norm"] = float(
            reward_out.detach().float().abs().sum().item()
        )
    if gate is not None and isinstance(gate, torch.Tensor):
        gate_f = gate.detach().float()
        metrics["parallel_reward/gate_mean"] = float(gate_f.mean().item())
        metrics["parallel_reward/gate_max"] = float(gate_f.max().item())
        if reward_out is not None and isinstance(reward_out, torch.Tensor):
            effective = gate_f * reward_out.detach().float()
            metrics["parallel_reward/branch_norm"] = float(effective.abs().sum().item())
            main_norm = metrics.get("parallel_reward/main_norm", 0.0)
            eps = 1e-8
            metrics["parallel_reward/branch_ratio"] = metrics[
                "parallel_reward/branch_norm"
            ] / (main_norm + eps)

    # Cache state from the first active layer.
    layer = active_layers[0]
    if past_key_values is not None and hasattr(past_key_values, "layers"):
        try:
            layer_idx = getattr(layer, "layer_idx", None)
            if layer_idx is None:
                layer_idx = layers.index(layer)
            lc = past_key_values.layers[layer_idx]
            baseline = getattr(lc, "reward_value_baseline", None)
            if baseline is not None:
                metrics["parallel_reward/value_baseline"] = float(
                    baseline.detach().float().abs().sum().item()
                )
            rec = getattr(lc, "reward_recurrent_state", None)
            if rec is not None:
                metrics["parallel_reward/recurrent_state_norm"] = _norm(rec)
        except (AttributeError, IndexError):
            pass

    return metrics


def maybe_warn_branch_ratio(
    metrics: dict[str, float],
    threshold: float,
) -> str | None:
    r"""Return a warning string when the branch contribution dominates the main path."""
    ratio = metrics.get("parallel_reward/branch_ratio")
    if ratio is None or threshold <= 0.0:
        return None
    if ratio > threshold:
        return (
            f"parallel reward branch contributes {ratio:.3f} of main-path norm "
            f"(threshold {threshold:.3f}). The gate may be growing too fast; "
            "consider reducing reward_gate_init_bias toward -7.0 or freezing "
            "reward_gate_proj for the first N steps."
        )
    return None
