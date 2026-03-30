from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.application.services.guidance_service import GuidanceService
from api.dependencies import get_guidance_service
from shared.contracts.inference import ApiGuidanceJobStatus, GuidanceRequest, JobAcceptedResponse

router = APIRouter(tags=["guidance"])


@router.post("/guidance/jobs", response_model=JobAcceptedResponse)
async def create_guidance_job(
    http_request: Request,
    request: GuidanceRequest,
    service: GuidanceService = Depends(get_guidance_service),
) -> JobAcceptedResponse:
    accepted = await service.submit_job(request)
    accepted.status_url = str(http_request.url_for("get_guidance_job_status", job_id=accepted.job_id))
    return accepted


@router.get(
    "/guidance/jobs/{job_id}",
    name="get_guidance_job_status",
    response_model=ApiGuidanceJobStatus,
    response_model_exclude_none=True,
    response_model_exclude_defaults=True,
)
async def get_guidance_job_status(
    job_id: str,
    service: GuidanceService = Depends(get_guidance_service),
) -> ApiGuidanceJobStatus:
    return await service.get_job_status(job_id)
