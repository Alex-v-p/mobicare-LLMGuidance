from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.application.services.ingestion_service import IngestionService
from api.dependencies import get_ingestion_service
from shared.contracts.ingestion import ApiIngestionJobStatus, IngestDocumentsRequest, IngestionCollectionDeleteResponse, IngestionJobAcceptedResponse

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


@router.get(
    "/ingestion/jobs/{job_id}",
    name="get_ingestion_job_status",
    response_model=ApiIngestionJobStatus,
    response_model_exclude_none=True,
    response_model_exclude_defaults=True,
)
async def get_ingestion_job_status(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiIngestionJobStatus:
    return await service.get_job_status(job_id)


@router.delete("/ingestion/collection", response_model=IngestionCollectionDeleteResponse)
async def delete_ingestion_collection(
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionCollectionDeleteResponse:
    return await service.delete_collection()
