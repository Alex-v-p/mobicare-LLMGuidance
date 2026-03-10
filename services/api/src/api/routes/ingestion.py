from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.application.services.ingestion_service import IngestionService
from api.clients.inference_client import InferenceClientError
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionJobAcceptedResponse, IngestionJobRecord

router = APIRouter(tags=["ingestion"])


@router.post("/ingestion/jobs", response_model=IngestionJobAcceptedResponse, status_code=202)
async def create_ingestion_job(
    http_request: Request,
    payload: IngestDocumentsRequest,
) -> IngestionJobAcceptedResponse:
    try:
        accepted = await IngestionService().submit_job(payload)
    except InferenceClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    accepted.status_url = str(http_request.url_for("get_ingestion_job_status", job_id=accepted.job_id))
    return accepted


@router.get("/ingestion/jobs/{job_id}", name="get_ingestion_job_status", response_model=IngestionJobRecord)
async def get_ingestion_job_status(job_id: str) -> IngestionJobRecord:
    try:
        return await IngestionService().get_job_status(job_id)
    except InferenceClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
