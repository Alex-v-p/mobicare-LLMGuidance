from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from inference.jobstore.redis_ingestion_store import RedisIngestionJobStore
from inference.storage.minio_ingestion_results import MinioIngestionResultStore
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionJobAcceptedResponse, IngestionJobRecord

router = APIRouter(tags=["ingestion"])


@router.post("/ingestion/jobs", response_model=IngestionJobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_ingestion_job(_: IngestDocumentsRequest | None = None, http_request: Request = None) -> IngestionJobAcceptedResponse:
    store = RedisIngestionJobStore()
    record = IngestionJobRecord(status="queued")
    try:
        await store.create(record)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        await store.close()

    return IngestionJobAcceptedResponse(
        job_id=record.job_id,
        status_url=str(http_request.url_for("get_ingestion_job_status_internal", job_id=record.job_id)),
    )


@router.get("/ingestion/jobs/{job_id}", name="get_ingestion_job_status_internal", response_model=IngestionJobRecord)
async def get_ingestion_job_status(job_id: str) -> IngestionJobRecord:
    store = RedisIngestionJobStore()
    try:
        record = await store.get(job_id)
    finally:
        await store.close()

    if record is None:
        raise HTTPException(status_code=404, detail=f"Ingestion job {job_id} was not found")

    if record.result is None and record.result_object_key:
        try:
            record = MinioIngestionResultStore().get_job_result(record.result_object_key)
        except Exception:
            pass
    return record
