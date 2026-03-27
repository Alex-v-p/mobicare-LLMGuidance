from __future__ import annotations

import pytest

from inference.application.services.guidance_service import GuidanceJobService, GuidanceRequestService
from inference.application.services.ingestion_service import IngestionJobService
from inference.infrastructure.http.exceptions import ConflictError
from shared.contracts.inference import InferenceRequest, InferenceResponse
from shared.contracts.ingestion import IngestDocumentsRequest
from tests.support.fakes import InMemoryGuidanceJobStore, InMemoryIngestionJobStore, InMemoryJobResultStore


class StaticRetrievalState:
    def __init__(self, ready: bool = True, state: str = "ready") -> None:
        self.ready = ready
        self.state = state
        self.calls: list[str] = []

    async def ensure_guidance_ready(self):
        self.calls.append("ensure")
        if not self.ready:
            raise ConflictError("Guidance is unavailable.", details={"retrieval_state": self.state})
        return None


class StaticGuidancePipeline:
    def __init__(self, response: InferenceResponse) -> None:
        self.response = response

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        return self.response.model_copy(update={"request_id": request.request_id})


@pytest.mark.asyncio
async def test_guidance_job_creation_is_blocked_when_retrieval_not_ready(inference_response):
    retrieval_state = StaticRetrievalState(ready=False, state="empty")
    service = GuidanceJobService(
        store_factory=InMemoryGuidanceJobStore,
        result_store=InMemoryJobResultStore(),
        retrieval_state=retrieval_state,
    )

    with pytest.raises(ConflictError):
        await service.create(InferenceRequest(request_id="req-1", question="question", patient_variables={}))

    assert retrieval_state.calls == ["ensure"]


@pytest.mark.asyncio
async def test_guidance_request_is_blocked_when_retrieval_not_ready(inference_response):
    retrieval_state = StaticRetrievalState(ready=False, state="ingesting")
    service = GuidanceRequestService(
        pipeline=StaticGuidancePipeline(inference_response),
        retrieval_state=retrieval_state,
    )

    with pytest.raises(ConflictError):
        await service.generate(InferenceRequest(request_id="req-1", question="question", patient_variables={}))

    assert retrieval_state.calls == ["ensure"]


@pytest.mark.asyncio
async def test_ingestion_job_creation_rejects_second_active_job():
    store = InMemoryIngestionJobStore()
    service = IngestionJobService(
        store_factory=lambda: store,
        result_store=InMemoryJobResultStore(),
    )

    await service.create(IngestDocumentsRequest())

    with pytest.raises(ConflictError):
        await service.create(IngestDocumentsRequest())
