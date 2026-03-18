from __future__ import annotations

from statistics import mean
from typing import Any



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



def summarize_stage_latencies(stage_lists: list[list[dict[str, Any]]]) -> dict[str, dict[str, float | int]]:
    values_by_stage: dict[str, list[float]] = {}
    status_counts: dict[str, dict[str, int]] = {}
    for stages in stage_lists:
        for stage in stages or []:
            if not isinstance(stage, dict):
                continue
            name = str(stage.get("name") or "unknown")
            status = str(stage.get("status") or "unknown")
            status_counts.setdefault(name, {})[status] = status_counts.setdefault(name, {}).get(status, 0) + 1
            duration = stage.get("duration_ms")
            if duration is None:
                continue
            try:
                values_by_stage.setdefault(name, []).append(float(duration) / 1000.0)
            except (TypeError, ValueError):
                continue

    summary: dict[str, dict[str, float | int]] = {}
    stage_names = sorted(set(values_by_stage) | set(status_counts))
    for name in stage_names:
        item: dict[str, float | int] = {
            "count": sum(status_counts.get(name, {}).values()),
            "completed_count": status_counts.get(name, {}).get("completed", 0),
        }
        item.update(summarize_latencies(values_by_stage.get(name, [])))
        summary[name] = item
    return summary
