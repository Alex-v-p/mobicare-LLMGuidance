from __future__ import annotations

from collections import defaultdict
from threading import Lock


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._hist_sums: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._hist_counts: defaultdict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)

    def _key(self, name: str, labels: dict[str, str] | None = None) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted((labels or {}).items()))

    def inc(self, name: str, value: float = 1.0, *, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += value

    def set_gauge(self, name: str, value: float, *, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = value

    def observe(self, name: str, value: float, *, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            key = self._key(name, labels)
            self._hist_sums[key] += value
            self._hist_counts[key] += 1

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(_render_metric(name, value, labels))
            for (name, labels), value in sorted(self._gauges.items()):
                lines.append(_render_metric(name, value, labels))
            for (name, labels), value in sorted(self._hist_sums.items()):
                lines.append(_render_metric(f"{name}_sum", value, labels))
            for (name, labels), value in sorted(self._hist_counts.items()):
                lines.append(_render_metric(f"{name}_count", value, labels))
        return "\n".join(lines) + "\n"


def _render_metric(name: str, value: float | int, labels: tuple[tuple[str, str], ...]) -> str:
    if labels:
        rendered = ",".join(f'{k}="{v}"' for k, v in labels)
        return f"{name}{{{rendered}}} {value}"
    return f"{name} {value}"


_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    return _registry
