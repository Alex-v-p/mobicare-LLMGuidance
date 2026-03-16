from __future__ import annotations

import pytest

from inference.http.exceptions import NotFoundError
from inference.http.services.guidance_service import GuidanceJobService
from shared.contracts.inference import InferenceRequest, InferenceResponse, JobRecord
from tests.support.fakes import InMemoryGuidanceJobStore, InMemoryJobResultStore


@pytest.mark.asyncio
async def test_get_loads_archived_result_when_record_has_object_key(inference_response):
    store = InMemoryGuidanceJobStore()
    result_store = InMemoryJobResultStore()
    service = GuidanceJobService(store_factory=lambda: store, result_store=result_store)
    request = InferenceRequest(request_id="req-1", question="question", patient_variables={})
    archived = JobRecord(request_id="req-1", status="completed", request=request, result=inference_response)
    object_key = result_store.put_job_result(archived)
    await store.create(JobRecord(request_id="req-1", status="completed", request=request, result_object_key=object_key))

    record = await service.get(next(iter(store.records)))

    assert record.result is not None
    assert record.result.answer == inference_response.answer


@pytest.mark.asyncio
async def test_get_raises_not_found_for_missing_job():
    service = GuidanceJobService(store_factory=InMemoryGuidanceJobStore, result_store=InMemoryJobResultStore())

    with pytest.raises(NotFoundError):
        await service.get("missing-job")
