"""Tests for the parallel reward monitoring helpers."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from qwendopamine.models.infinidopamine import (
    InfiniDopamineForCausalLM,
    InfiniDopamineTextConfig,
)
from qwendopamine.training import (
    collect_parallel_reward_metrics,
    maybe_warn_branch_ratio,
)


def _build(use_pr: bool) -> InfiniDopamineForCausalLM:
    return InfiniDopamineForCausalLM(
        InfiniDopamineTextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            vocab_size=100,
            layer_types=["linear_attention", "full_attention"],
            use_parallel_reward=use_pr,
        )
    )


def test_collect_metrics_empty_when_no_parallel_branch() -> None:
    m = _build(use_pr=False)
    metrics = collect_parallel_reward_metrics(m)
    assert metrics == {"parallel_reward/active_layers": 0.0}


def test_collect_metrics_reports_active_layers() -> None:
    m = _build(use_pr=True)
    metrics = collect_parallel_reward_metrics(m)
    assert metrics.get("parallel_reward/active_layers") == 1.0


def test_collect_metrics_reads_cache_state() -> None:
    m = _build(use_pr=True)
    m.eval()
    cache = DynamicCache(config=m.config)
    rewards = torch.ones(1, 4, 6) * 1.5
    with torch.no_grad():
        m(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            past_key_values=cache,
            reward_values=rewards,
            use_cache=True,
        )
    metrics = collect_parallel_reward_metrics(m, past_key_values=cache)
    assert "parallel_reward/value_baseline" in metrics, metrics
    assert metrics["parallel_reward/value_baseline"] > 0.0


def test_maybe_warn_branch_ratio_returns_warning() -> None:
    assert (
        maybe_warn_branch_ratio({"parallel_reward/branch_ratio": 0.5}, 0.1) is not None
    )
    assert maybe_warn_branch_ratio({"parallel_reward/branch_ratio": 0.01}, 0.1) is None
    assert maybe_warn_branch_ratio({}, 0.1) is None
    assert maybe_warn_branch_ratio({"parallel_reward/branch_ratio": 0.5}, 0.0) is None
