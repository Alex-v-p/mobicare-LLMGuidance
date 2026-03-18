from __future__ import annotations

from statistics import mean


def summarize_latencies(values: list[float]) -> dict[str, float]:
    if not values:
        return {"average": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)
    def pct(p: float) -> float:
        index = int(round((len(ordered) - 1) * p))
        return ordered[index]
    return {
        "average": mean(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
    }
