from __future__ import annotations

from fastapi import APIRouter, Request

from api.application.services.guidance_service import GuidanceService
from shared.contracts.inference import ApiGuidanceJobStatus, ApiGuidanceResponse, GuidanceRequest, JobAcceptedResponse
from shared.contracts.ingestion import IngestionResponse

router = APIRouter(tags=["guidance"])


@router.post("/guidance", response_model=ApiGuidanceResponse)
async def generate_guidance(request: GuidanceRequest) -> ApiGuidanceResponse:
    return await GuidanceService().generate(request)


@router.post("/guidance/ingest", response_model=IngestionResponse)
async def ingest_guidance_documents() -> IngestionResponse:
    return await GuidanceService().ingest()


@router.post("/guidance/jobs", response_model=JobAcceptedResponse, status_code=202)
async def create_guidance_job(request: GuidanceRequest, http_request: Request) -> JobAcceptedResponse:
    accepted = await GuidanceService().submit_job(request)
    accepted.status_url = str(http_request.url_for("get_guidance_job_status", job_id=accepted.job_id))
    return accepted


@router.get("/guidance/jobs/{job_id}", name="get_guidance_job_status", response_model=ApiGuidanceJobStatus)
async def get_guidance_job_status(job_id: str) -> ApiGuidanceJobStatus:
    return await GuidanceService().get_job_status(job_id)
