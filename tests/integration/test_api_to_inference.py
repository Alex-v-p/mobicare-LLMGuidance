from __future__ import annotations

import httpx
import pytest

from api.main import create_app as create_api_app
from inference.http.dependencies import (
    get_guidance_job_service,
    get_guidance_request_service,
    get_ingestion_job_service,
    get_ingestion_request_service,
)
from inference.http.main import create_app as create_inference_app
from shared.contracts.inference import JobRecord
from shared.contracts.ingestion import IngestionCollectionDeleteResponse, IngestionJobRecord
from tests.support.fakes import (
    DummyAsyncClientContext,
    InMemoryGuidanceJobStore,
    InMemoryIngestionJobStore,
    StaticGuidancePipeline,
    StaticIngestionService,
)


class StubGuidanceRequestService:
    def __init__(self, pipeline: StaticGuidancePipeline) -> None:
        self._pipeline = pipeline

    async def generate(self, request):
        return await self._pipeline.run(request)


class StubGuidanceJobService:
    def __init__(self, store: InMemoryGuidanceJobStore) -> None:
        self._store = store

    async def create(self, request):
        record = JobRecord(request_id=request.request_id, status="queued", request=request)
        await self._store.create(record)
        return record

    async def get(self, job_id: str):
        return await self._store.get(job_id)


class StubIngestionRequestService:
    def __init__(self, service: StaticIngestionService) -> None:
        self._service = service

    async def ingest(self, payload):
        return await self._service.ingest(payload)

    async def delete_collection(self):
        return IngestionCollectionDeleteResponse(collection="guidance_chunks", existed=True)


class StubIngestionJobService:
    def __init__(self, store: InMemoryIngestionJobStore) -> None:
        self._store = store

    async def create(self, payload):
        record = IngestionJobRecord(status="queued", request=payload)
        await self._store.create(record)
        return record

    async def get(self, job_id: str):
        return await self._store.get(job_id)


@pytest.mark.asyncio
async def test_api_proxies_guidance_and_ingestion_requests(guidance_request, inference_response, ingestion_response, monkeypatch):
    inference_app = create_inference_app()
    guidance_store = InMemoryGuidanceJobStore()
    ingestion_store = InMemoryIngestionJobStore()

    inference_app.dependency_overrides[get_guidance_request_service] = lambda: StubGuidanceRequestService(
        StaticGuidancePipeline(inference_response)
    )
    inference_app.dependency_overrides[get_guidance_job_service] = lambda: StubGuidanceJobService(guidance_store)
    inference_app.dependency_overrides[get_ingestion_request_service] = lambda: StubIngestionRequestService(
        StaticIngestionService(ingestion_response)
    )
    inference_app.dependency_overrides[get_ingestion_job_service] = lambda: StubIngestionJobService(ingestion_store)

    monkeypatch.setattr(
        "api.clients.inference_client.create_async_client",
        lambda timeout_s: DummyAsyncClientContext(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=inference_app),
                base_url="http://inference.test",
                timeout=timeout_s,
            )
        ),
    )

    api_app = create_api_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://api.test") as client:
        guidance_job = await client.post("/guidance/jobs", json=guidance_request.model_dump(mode="json"))
        ingestion_job = await client.post("/ingestion/jobs", json={"options": {}})
        delete_resp = await client.delete("/ingestion/collection")

    assert guidance_job.status_code == 200
    assert guidance_job.json()["job_id"]
    assert guidance_job.json()["request_id"] == guidance_request.request_id
    assert ingestion_job.status_code == 202
    assert ingestion_job.json()["status_url"].endswith(f"/ingestion/jobs/{ingestion_job.json()['job_id']}")
    assert delete_resp.json()["collection"] == "guidance_chunks"
