from __future__ import annotations


class MetricTracker:
    r"""Simple metric accumulator.

    Attributes:
        values (dict[str, float]): latest metric values.
    """

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def update(self, name: str, value: float) -> None:
        r"""Record a metric value.

        Args:
            name (str): metric name.
            value (float): metric value.
        """
        self.values[name] = value

    def state_dict(self) -> dict[str, float]:
        r"""Return a copy of the tracked metrics.

        Returns:
            dict[str, float]: metric values.
        """
        return dict(self.values)
