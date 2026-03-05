from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    """Result of checking a single dependency endpoint."""

    ok: bool
    url: str
    status_code: Optional[int] = None
    error: Optional[str] = None


class HealthReport(BaseModel):
    """Aggregated health report returned by the API."""

    status: str  # "ok" | "degraded"
    deps: Dict[str, DependencyStatus]
