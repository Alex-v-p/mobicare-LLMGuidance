from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from inference.indexing.ingestion_service import IngestionService
from inference.jobstore.redis_ingestion_job_store import RedisIngestionJobStore
from inference.storage.minio_ingestion_job_results import MinioIngestionJobResultStore
from shared.contracts.ingestion import (
    IngestDocumentsRequest,
    IngestionJobAcceptedResponse,
    IngestionJobRecord,
    IngestionResponse,
)

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(payload: IngestDocumentsRequest) -> IngestionResponse:
    service = IngestionService()
    return await service.ingest(payload)


@router.post(
    "/ingestion/jobs",
    response_model=IngestionJobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_ingestion_job(
    http_request: Request,
    payload: IngestDocumentsRequest,
) -> IngestionJobAcceptedResponse:
    store = RedisIngestionJobStore()
    record = IngestionJobRecord(status="queued", request=payload)

    try:
        await store.create(record)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        await store.close()

    return IngestionJobAcceptedResponse(
        job_id=record.job_id,
        status_url=str(http_request.url_for("get_ingestion_job_status", job_id=record.job_id)),
    )


@router.get(
    "/ingestion/jobs/{job_id}",
    name="get_ingestion_job_status",
    response_model=IngestionJobRecord,
)
async def get_ingestion_job_status(job_id: str) -> IngestionJobRecord:
    store = RedisIngestionJobStore()

    try:
        record = await store.get(job_id)
    finally:
        await store.close()

    if record is None:
        raise HTTPException(status_code=404, detail=f"Ingestion job {job_id} was not found")

    if record.result is None and record.result_object_key:
        result_store = MinioIngestionJobResultStore()
        try:
            record = await result_store.get_job_result(record.result_object_key)
        except Exception:
            pass

    return record
