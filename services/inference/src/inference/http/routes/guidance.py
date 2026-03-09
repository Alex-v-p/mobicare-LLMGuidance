from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from inference.jobstore.redis_guidance_job_store import RedisGuidanceJobStore
from inference.pipeline.generate_guidance import GuidancePipeline
from inference.storage.minio_guidance_job_results import MinioGuidanceJobResultStore
from shared.contracts.inference import InferenceRequest, InferenceResponse, JobAcceptedResponse, JobRecord

router = APIRouter(tags=["guidance"])


@router.post("/guidance/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest) -> InferenceResponse:
    return await GuidancePipeline().run(request)


@router.post("/guidance/jobs", response_model=JobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: InferenceRequest, http_request: Request) -> JobAcceptedResponse:
    store = RedisGuidanceJobStore()
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
        status_url=str(http_request.url_for("get_guidance_job_status", job_id=record.job_id)),
    )


@router.get("/guidance/jobs/{job_id}", name="get_guidance_job_status", response_model=JobRecord)
async def get_job_status(job_id: str) -> JobRecord:
    store = RedisGuidanceJobStore()
    try:
        record = await store.get(job_id)
    finally:
        await store.close()

    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} was not found")

    if record.result is None and record.result_object_key:
        try:
            record = MinioGuidanceJobResultStore().get_job_result(record.result_object_key)
        except Exception:
            pass
    return record
