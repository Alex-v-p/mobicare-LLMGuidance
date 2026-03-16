from __future__ import annotations

import pytest

from inference.http.exceptions import NotFoundError
from inference.http.services.ingestion_service import IngestionJobService, IngestionRequestService
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionJobRecord, IngestionResponse
from tests.support.fakes import InMemoryIngestionJobStore, InMemoryJobResultStore


class StaticVectorStore:
    def __init__(self, existed: bool = True) -> None:
        self.collection_name = "guidance_chunks"
        self.existed = existed

    def delete_collection(self) -> bool:
        return self.existed


class StaticIngestionBackend:
    def __init__(self, response: IngestionResponse) -> None:
        self.response = response

    async def ingest(self, payload: IngestDocumentsRequest) -> IngestionResponse:
        return self.response


@pytest.mark.asyncio
async def test_ingestion_request_service_delegates_ingest():
    response = IngestionResponse(documents_bucket="docs", collection="guidance_chunks")
    service = IngestionRequestService(ingestion_service=StaticIngestionBackend(response), vector_store=StaticVectorStore())

    result = await service.ingest(IngestDocumentsRequest())

    assert result == response


@pytest.mark.asyncio
async def test_delete_collection_returns_status_payload():
    service = IngestionRequestService(ingestion_service=StaticIngestionBackend(IngestionResponse(documents_bucket="docs", collection="guidance_chunks")), vector_store=StaticVectorStore(existed=False))

    result = await service.delete_collection()

    assert result.collection == "guidance_chunks"
    assert result.existed is False


@pytest.mark.asyncio
async def test_ingestion_job_service_create_and_get_round_trip():
    store = InMemoryIngestionJobStore()
    service = IngestionJobService(store_factory=lambda: store, result_store=InMemoryJobResultStore())

    created = await service.create(IngestDocumentsRequest())
    loaded = await service.get(created.job_id)

    assert loaded.job_id == created.job_id
    assert loaded.status == "queued"


@pytest.mark.asyncio
async def test_ingestion_job_service_raises_not_found_for_missing_job():
    service = IngestionJobService(store_factory=InMemoryIngestionJobStore, result_store=InMemoryJobResultStore())

    with pytest.raises(NotFoundError):
        await service.get("missing")


@pytest.mark.asyncio
async def test_ingestion_job_service_loads_archived_result():
    store = InMemoryIngestionJobStore()
    results = InMemoryJobResultStore()
    service = IngestionJobService(store_factory=lambda: store, result_store=results)
    record = IngestionJobRecord(status="completed", result=None, result_object_key="jobs/result-1.json")
    results.records["jobs/result-1.json"] = record.model_copy(update={"result": IngestionResponse(documents_bucket="docs", collection="guidance_chunks")})
    await store.create(record.model_copy(update={"status": "queued"}))
    await store.update(record)

    loaded = await service.get(record.job_id)

    assert loaded.result is not None
    assert loaded.result.collection == "guidance_chunks"
