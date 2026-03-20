from __future__ import annotations

from datetime import datetime, timezone



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    return datetime.fromisoformat(candidate)



def duration_seconds(started_at: str | None, completed_at: str | None) -> float | None:
    start_dt = parse_iso_datetime(started_at)
    end_dt = parse_iso_datetime(completed_at)
    if not start_dt or not end_dt:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds())
