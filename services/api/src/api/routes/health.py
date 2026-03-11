from __future__ import annotations

from fastapi import APIRouter, Depends

from api.application.services.health_service import HealthService
from shared.contracts.health import HealthReport

router = APIRouter(tags=["health"])


def get_health_service() -> HealthService:
    return HealthService()


@router.get("/health", response_model=HealthReport)
async def get_health(
    service: HealthService = Depends(get_health_service),
) -> HealthReport:
    return await service.report()
