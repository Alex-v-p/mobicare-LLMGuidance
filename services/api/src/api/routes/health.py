from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from api.application.services.health_service import HealthService
from api.dependencies import get_health_service
from shared.contracts.health import HealthReport
from shared.observability import get_metrics_registry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthReport)
async def get_health(service: HealthService = Depends(get_health_service)) -> HealthReport:
    return await service.report()


@router.get("/metrics", response_class=PlainTextResponse)
def get_metrics() -> PlainTextResponse:
    return PlainTextResponse(get_metrics_registry().render_prometheus(), media_type="text/plain; version=0.0.4")
