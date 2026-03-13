from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.application.services.ingestion_service import IngestionService
from api.dependencies import get_ingestion_service
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionJobAcceptedResponse, IngestionJobRecord

router = APIRouter(tags=["ingestion"])


@router.post("/ingestion/jobs", response_model=IngestionJobAcceptedResponse, status_code=202)
async def create_ingestion_job(
    http_request: Request,
    payload: IngestDocumentsRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionJobAcceptedResponse:
    accepted = await service.submit_job(payload)
    accepted.status_url = str(http_request.url_for("get_ingestion_job_status", job_id=accepted.job_id))
    return accepted


@router.get("/ingestion/jobs/{job_id}", name="get_ingestion_job_status", response_model=IngestionJobRecord)
async def get_ingestion_job_status(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionJobRecord:
    return await service.get_job_status(job_id)
