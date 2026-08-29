r"""Metric tracking and logging accumulators."""

from __future__ import annotations


class MetricTracker:
    r"""Accumulate scalar training metrics with running averages and history tracking."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.history: dict[str, list[float]] = {}
        self.running_sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, name: str, value: float) -> None:
        r"""Record a scalar metric value under the given name."""
        val = float(value)
        self.values[name] = val
        if name not in self.history:
            self.history[name] = []
            self.running_sums[name] = 0.0
            self.counts[name] = 0
        self.history[name].append(val)
        self.running_sums[name] += val
        self.counts[name] += 1

    def get_mean(self, name: str) -> float:
        r"""Return running average for a metric, or 0.0 if not recorded."""
        count = self.counts.get(name, 0)
        if count == 0:
            return 0.0
        return self.running_sums[name] / count

    def get_history(self, name: str) -> list[float]:
        r"""Return a copy of the recorded history for a metric."""
        return list(self.history.get(name, []))

    def reset(self) -> None:
        r"""Clear all recorded metric values, running sums, and history."""
        self.values.clear()
        self.history.clear()
        self.running_sums.clear()
        self.counts.clear()

    def state_dict(self) -> dict[str, float]:
        r"""Return a dictionary copy of latest recorded metric values."""
        return dict(self.values)

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        r"""Restore metric values from a dictionary."""
        for k, v in state_dict.items():
            self.update(k, v)


__all__ = ["MetricTracker"]
