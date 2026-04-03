from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Any


def summarize_latencies(values: list[float], *, policy: str = "all", outlier_policy: str = "keep_all") -> dict[str, float | int | str]:
    if outlier_policy == "exclude_failures":
        values = [value for value in values if value is not None]
    if not values:
        return {
            "average": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "count": 0,
            "included_count": 0,
            "policy": policy,
            "outlier_policy": outlier_policy,
        }
    ordered = sorted(float(value) for value in values)

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
        "count": len(values),
        "included_count": len(ordered),
        "policy": policy,
        "outlier_policy": outlier_policy,
    }


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _stage_duration_seconds(stage: dict[str, Any]) -> float | None:
    duration_ms = stage.get("duration_ms")
    if duration_ms is not None:
        try:
            return float(duration_ms) / 1000.0
        except (TypeError, ValueError):
            return None
    started_at = _parse_iso(stage.get("started_at"))
    completed_at = _parse_iso(stage.get("completed_at"))
    if started_at is None or completed_at is None:
        return None
    return max(0.0, (completed_at - started_at).total_seconds())


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
            duration_seconds = _stage_duration_seconds(stage)
            if duration_seconds is None:
                continue
            values_by_stage.setdefault(name, []).append(duration_seconds)

    summary: dict[str, dict[str, float | int]] = {}
    stage_names = sorted(set(values_by_stage) | set(status_counts))
    for name in stage_names:
        event_count = sum(status_counts.get(name, {}).values())
        timed_values = values_by_stage.get(name, [])
        duration_summary = summarize_latencies(timed_values)
        summary[name] = {
            "count": event_count,
            "completed_count": status_counts.get(name, {}).get("completed", 0),
            "timed_count": len(timed_values),
            "untimed_count": max(0, event_count - len(timed_values)),
            "average": duration_summary.get("average", 0.0),
            "min": duration_summary.get("min", 0.0),
            "max": duration_summary.get("max", 0.0),
            "p50": duration_summary.get("p50", 0.0),
            "p95": duration_summary.get("p95", 0.0),
            "p99": duration_summary.get("p99", 0.0),
            "included_count": duration_summary.get("included_count", 0),
            "policy": "all",
            "outlier_policy": "keep_all",
        }
    return summary
