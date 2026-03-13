from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.application.services.guidance_service import GuidanceService
from api.dependencies import get_guidance_service
from api.clients.inference_client import InferenceClientError
from shared.contracts.inference import (
    ApiGuidanceJobStatus,
    ApiGuidanceResponse,
    GuidanceRequest,
    JobAcceptedResponse,
)

router = APIRouter(tags=["guidance"])


@router.post("/guidance/generate", response_model=ApiGuidanceResponse)
async def generate_guidance(
    request: GuidanceRequest,
    service: GuidanceService = Depends(get_guidance_service),
) -> ApiGuidanceResponse:
    try:
        return await service.generate(request)
    except InferenceClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/guidance/jobs", response_model=JobAcceptedResponse)
async def create_guidance_job(
    request: GuidanceRequest,
    service: GuidanceService = Depends(get_guidance_service),
) -> JobAcceptedResponse:
    try:
        return await service.submit_job(request)
    except InferenceClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/guidance/jobs/{job_id}", response_model=ApiGuidanceJobStatus)
async def get_guidance_job_status(
    job_id: str,
    service: GuidanceService = Depends(get_guidance_service),
) -> ApiGuidanceJobStatus:
    try:
        return await service.get_job_status(job_id)
    except InferenceClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
