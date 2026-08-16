r"""Metric tracking and logging accumulators."""

from __future__ import annotations


class MetricTracker:
    r"""MetricTracker()

    Accumulates scalar training metrics and diagnostic statistics into a key-value store.

    Examples::

        >>> tracker = MetricTracker()
        >>> tracker.update("loss", 0.25)
        >>> tracker.state_dict()
        {'loss': 0.25}
    """

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def update(self, name: str, value: float) -> None:
        r"""update(name, value) -> None

        Records a scalar metric value under the specified metric name.

        Args:
            name (str): Unique metric identifier string.
            value (float): Metric numerical value.
        """
        self.values[name] = value

    def state_dict(self) -> dict[str, float]:
        r"""state_dict() -> dict[str, float]

        Returns a dictionary copy of all recorded metric values.

        Returns:
            dict[str, float]: Dictionary mapping metric names to latest recorded values.
        """
        return dict(self.values)


__all__ = ["MetricTracker"]
