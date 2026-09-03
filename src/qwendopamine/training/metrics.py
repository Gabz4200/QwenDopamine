r"""Metric tracking and logging accumulators."""

from __future__ import annotations


class MetricTracker:
    r"""MetricTracker() -> None

    Accumulate scalar training metrics with running averages and history.

    Attributes:
        values (dict[str, float]): Latest recorded value per metric.
        history (dict[str, list[float]]): Full history per metric.
        running_sums (dict[str, float]): Cumulative sum per metric.
        counts (dict[str, int]): Sample count per metric.
    """

    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.history: dict[str, list[float]] = {}
        self.running_sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, name: str, value: float) -> None:
        r"""update(name: str, value: float) -> None

        Record a scalar metric value under the given name.

        Args:
            name (str): Metric key.
            value (float): Observed value.

        Returns:
            None
        """
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
        r"""get_mean(name: str) -> float

        Return running average for a metric, or 0.0 if not recorded.

        Args:
            name (str): Metric key.

        Returns:
            float: Current running average.
        """
        count = self.counts.get(name, 0)
        if count == 0:
            return 0.0
        return self.running_sums[name] / count

    def get_history(self, name: str) -> list[float]:
        r"""get_history(name: str) -> list[float]

        Return a copy of the recorded history for a metric.

        Args:
            name (str): Metric key.

        Returns:
            list[float]: Copy of the history (empty if no data).
        """
        return list(self.history.get(name, []))

    def reset(self) -> None:
        r"""reset() -> None

        Clear all recorded metric values, running sums, and history.

        Returns:
            None
        """
        self.values.clear()
        self.history.clear()
        self.running_sums.clear()
        self.counts.clear()

    def state_dict(self) -> dict[str, float]:
        r"""state_dict() -> dict[str, float]

        Return a dictionary copy of latest recorded metric values.

        Returns:
            dict[str, float]: Latest value per metric.
        """
        return dict(self.values)

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        r"""load_state_dict(state_dict: dict[str, float]) -> None

        Restore metric values from a dictionary.

        Args:
            state_dict (dict[str, float]): Latest values per metric.

        Returns:
            None
        """
        for k, v in state_dict.items():
            self.update(k, v)


__all__ = ["MetricTracker"]
