from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from inference.jobstore.redis_store import RedisJobStore
from inference.storage.minio_results import MinioResultStore
from shared.contracts.inference import InferenceRequest, JobAcceptedResponse, JobRecord

router = APIRouter(tags=["jobs"])


@router.post("/jobs", response_model=JobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: InferenceRequest, http_request: Request) -> JobAcceptedResponse:
    store = RedisJobStore()
    record = JobRecord(request_id=request.request_id, status="queued", request=request)
    try:
        await store.create(record)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        await store.close()

    return JobAcceptedResponse(
        job_id=record.job_id,
        request_id=request.request_id,
        status_url=str(http_request.url_for("get_job_status", job_id=record.job_id)),
    )


@router.get("/jobs/{job_id}", name="get_job_status", response_model=JobRecord)
async def get_job_status(job_id: str) -> JobRecord:
    store = RedisJobStore()
    try:
        record = await store.get(job_id)
    finally:
        await store.close()

    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} was not found")

    if record.result is None and record.result_object_key:
        try:
            record = MinioResultStore().get_job_result(record.result_object_key)
        except Exception:
            pass
    return record
