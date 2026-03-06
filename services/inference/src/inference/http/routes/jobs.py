from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from inference.jobstore import FileJobStore
from shared.contracts.inference import InferenceRequest, JobAcceptedResponse, JobRecord

router = APIRouter(tags=["jobs"])


@router.post("/jobs", response_model=JobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: InferenceRequest, http_request: Request) -> JobAcceptedResponse:
    store = FileJobStore()
    record = JobRecord(request_id=request.request_id, status="queued", request=request)
    try:
        store.enqueue(record)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return JobAcceptedResponse(
        request_id=request.request_id,
        status_url=str(http_request.url_for("get_job_status", request_id=request.request_id)),
    )


@router.get("/jobs/{request_id}", name="get_job_status", response_model=JobRecord)
async def get_job_status(request_id: str) -> JobRecord:
    store = FileJobStore()
    record = store.find(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {request_id} was not found")
    return record
