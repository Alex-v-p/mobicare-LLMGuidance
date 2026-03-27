from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from inference.infrastructure.http.dependencies import get_guidance_job_service, get_guidance_request_service
from inference.application.services.guidance_service import GuidanceJobService, GuidanceRequestService
from shared.contracts.inference import InferenceRequest, InferenceResponse, JobAcceptedResponse, JobRecord

router = APIRouter(tags=["guidance"])


@router.post("/guidance/generate", response_model=InferenceResponse)
async def generate(
    request: InferenceRequest,
    service: GuidanceRequestService = Depends(get_guidance_request_service),
) -> InferenceResponse:
    return await service.generate(request)


@router.post("/guidance/jobs", response_model=JobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: InferenceRequest,
    http_request: Request,
    service: GuidanceJobService = Depends(get_guidance_job_service),
) -> JobAcceptedResponse:
    record = await service.create(request)
    return JobAcceptedResponse(
        job_id=record.job_id,
        request_id=request.request_id,
        status_url=str(http_request.url_for("get_guidance_job_status", job_id=record.job_id)),
    )


@router.get("/guidance/jobs/{job_id}", name="get_guidance_job_status", response_model=JobRecord)
async def get_job_status(
    job_id: str,
    service: GuidanceJobService = Depends(get_guidance_job_service),
) -> JobRecord:
    return await service.get(job_id)
