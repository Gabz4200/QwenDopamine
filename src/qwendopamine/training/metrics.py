r"""Metric tracking and logging accumulators."""

from __future__ import annotations


class MetricTracker:
    r"""MetricTracker()

    Accumulates scalar training metrics and diagnostic statistics with running averages and history tracking.

    Examples::

        >>> tracker = MetricTracker()
        >>> tracker.update("loss", 0.5)
        >>> tracker.update("loss", 0.3)
        >>> tracker.get_mean("loss")
        0.4
        >>> tracker.state_dict()
        {'loss': 0.3}
    """

    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.history: dict[str, list[float]] = {}
        self.running_sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, name: str, value: float) -> None:
        r"""update(name, value) -> None

        Records a scalar metric value under the specified metric name.

        Args:
            name (str): Unique metric identifier string.
            value (float): Metric numerical value.
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
        r"""get_mean(name) -> float

        Returns running average value for the specified metric name.

        Args:
            name (str): Metric identifier string.

        Returns:
            float: Running mean, or 0.0 if not recorded.
        """
        count = self.counts.get(name, 0)
        if count == 0:
            return 0.0
        return self.running_sums[name] / count

    def get_history(self, name: str) -> list[float]:
        r"""get_history(name) -> list[float]

        Returns list of all recorded values for the specified metric name.

        Args:
            name (str): Metric identifier string.

        Returns:
            list[float]: Recorded history list copy.
        """
        return list(self.history.get(name, []))

    def reset(self) -> None:
        r"""reset() -> None

        Clears all recorded metric values, running sums, and history.
        """
        self.values.clear()
        self.history.clear()
        self.running_sums.clear()
        self.counts.clear()

    def state_dict(self) -> dict[str, float]:
        r"""state_dict() -> dict[str, float]

        Returns a dictionary copy of all recorded latest metric values.

        Returns:
            dict[str, float]: Dictionary mapping metric names to latest recorded values.
        """
        return dict(self.values)

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        r"""load_state_dict(state_dict) -> None

        Restores metric values from dictionary.

        Args:
            state_dict (dict[str, float]): Dictionary of metric values.
        """
        for k, v in state_dict.items():
            self.update(k, v)


__all__ = ["MetricTracker"]
