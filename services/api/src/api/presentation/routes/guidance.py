from __future__ import annotations

from fastapi import APIRouter, Depends

from api.application.services.guidance_service import GuidanceService
from api.dependencies import get_guidance_service
from shared.contracts.inference import ApiGuidanceJobStatus, GuidanceRequest, JobAcceptedResponse

router = APIRouter(tags=["guidance"])


@router.post("/guidance/jobs", response_model=JobAcceptedResponse)
async def create_guidance_job(
    request: GuidanceRequest,
    service: GuidanceService = Depends(get_guidance_service),
) -> JobAcceptedResponse:
    return await service.submit_job(request)


@router.get(
    "/guidance/jobs/{job_id}",
    response_model=ApiGuidanceJobStatus,
    response_model_exclude_none=True,
    response_model_exclude_defaults=True,
)
async def get_guidance_job_status(
    job_id: str,
    service: GuidanceService = Depends(get_guidance_service),
) -> ApiGuidanceJobStatus:
    return await service.get_job_status(job_id)
