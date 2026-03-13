from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from inference.http.dependencies import get_ingestion_job_service, get_ingestion_request_service
from inference.http.services.ingestion_service import IngestionJobService, IngestionRequestService
from shared.contracts.ingestion import (
    IngestDocumentsRequest,
    IngestionJobAcceptedResponse,
    IngestionJobRecord,
    IngestionResponse,
)

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(
    payload: IngestDocumentsRequest,
    service: IngestionRequestService = Depends(get_ingestion_request_service),
) -> IngestionResponse:
    return await service.ingest(payload)


@router.post(
    "/ingestion/jobs",
    response_model=IngestionJobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_ingestion_job(
    http_request: Request,
    payload: IngestDocumentsRequest,
    service: IngestionJobService = Depends(get_ingestion_job_service),
) -> IngestionJobAcceptedResponse:
    record = await service.create(payload)
    return IngestionJobAcceptedResponse(
        job_id=record.job_id,
        status_url=str(http_request.url_for("get_ingestion_job_status", job_id=record.job_id)),
    )


@router.get(
    "/ingestion/jobs/{job_id}",
    name="get_ingestion_job_status",
    response_model=IngestionJobRecord,
)
async def get_ingestion_job_status(
    job_id: str,
    service: IngestionJobService = Depends(get_ingestion_job_service),
) -> IngestionJobRecord:
    return await service.get(job_id)
