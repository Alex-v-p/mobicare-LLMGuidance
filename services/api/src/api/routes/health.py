from __future__ import annotations

from fastapi import APIRouter

from api.application.services.health_service import HealthService
from shared.contracts.health import HealthReport

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthReport)
async def health() -> HealthReport:
    """Single health endpoint.

    - Confirms the API process is alive.
    - Probes dependencies over HTTP and reports their status.
    """

    return await HealthService().report()