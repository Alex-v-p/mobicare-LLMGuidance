from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    ok: bool
    url: str
    status_code: Optional[int] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthReport(BaseModel):
    status: str
    deps: Dict[str, DependencyStatus]
