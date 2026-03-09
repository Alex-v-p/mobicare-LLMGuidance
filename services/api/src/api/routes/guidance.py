from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.application.services.guidance_service import GuidanceService
from api.clients.inference_client import InferenceClientError
from shared.contracts.inference import ApiGuidanceJobStatus, ApiGuidanceResponse, GuidanceRequest, JobAcceptedResponse

router = APIRouter(tags=["guidance"])


@router.post("/guidance", response_model=ApiGuidanceResponse)
async def generate_guidance(request: GuidanceRequest) -> ApiGuidanceResponse:
    try:
        return await GuidanceService().generate(request)
    except InferenceClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/guidance/jobs", response_model=JobAcceptedResponse, status_code=202)
async def create_guidance_job(request: GuidanceRequest, http_request: Request) -> JobAcceptedResponse:
    try:
        accepted = await GuidanceService().submit_job(request)
    except InferenceClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    accepted.status_url = str(http_request.url_for("get_guidance_job_status", job_id=accepted.job_id))
    return accepted


@router.get("/guidance/jobs/{job_id}", name="get_guidance_job_status", response_model=ApiGuidanceJobStatus)
async def get_guidance_job_status(job_id: str) -> ApiGuidanceJobStatus:
    try:
        return await GuidanceService().get_job_status(job_id)
    except InferenceClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
