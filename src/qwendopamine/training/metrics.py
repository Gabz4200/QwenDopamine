from __future__ import annotations


class MetricTracker:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def update(self, name: str, value: float) -> None:
        self.values[name] = value

    def state_dict(self) -> dict[str, float]:
        return dict(self.values)
