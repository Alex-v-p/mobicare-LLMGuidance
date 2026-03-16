from __future__ import annotations

from inference.worker.handlers.ingestion_handler import handle_ingestion_jobs
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionJobRecord, IngestionResponse

from tests.support.fakes import InMemoryJobResultStore


class InMemoryIngestionJobStore:
    def __init__(self) -> None:
        self.records = {}
        self.queue = []
        self.closed = False

    async def create(self, record):
        self.records[record.job_id] = record.model_copy(deep=True)
        self.queue.append(record.job_id)

    async def get(self, job_id):
        record = self.records.get(job_id)
        return record.model_copy(deep=True) if record is not None else None

    async def update(self, record):
        self.records[record.job_id] = record.model_copy(deep=True)

    async def close(self):
        self.closed = True

    async def claim_next(self, worker_id: str, timeout_s: int = 1):
        if not self.queue:
            return None
        job_id = self.queue.pop(0)
        record = self.records[job_id].model_copy(deep=True)
        record.status = "running"
        record.worker_id = worker_id
        record.lease_expires_at = "2026-03-16T10:05:00+00:00"
        self.records[job_id] = record.model_copy(deep=True)
        return record

    async def heartbeat(self, job_id: str, worker_id: str) -> bool:
        record = self.records.get(job_id)
        return bool(record and record.status == "running" and record.worker_id == worker_id)

    async def mark_completed(self, record, *, result, completed_at, result_object_key=None):
        record.status = "completed"
        record.result = result
        record.completed_at = completed_at
        record.result_object_key = result_object_key
        record.lease_expires_at = None
        self.records[record.job_id] = record.model_copy(deep=True)

    async def mark_failed(self, record, *, error, completed_at, result_object_key=None):
        record.status = "failed"
        record.error = error
        record.completed_at = completed_at
        record.result_object_key = result_object_key
        record.lease_expires_at = None
        self.records[record.job_id] = record.model_copy(deep=True)


class StaticIngestionService:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def ingest(self, payload):
        if self.error:
            raise self.error
        return self.response


async def test_handle_ingestion_jobs_completes_and_persists_result(monkeypatch):
    store = InMemoryIngestionJobStore()
    results = InMemoryJobResultStore()
    response = IngestionResponse(
        documents_bucket="docs",
        documents_prefix="guidelines",
        documents_found=1,
        chunks_created=2,
        vectors_upserted=2,
        collection="guidance",
        cleaning_strategy="basic",
        chunking_strategy="naive",
        cleaning_params={},
        chunking_params={},
        embedding_model="embed",
    )
    service = StaticIngestionService(response=response)
    record = IngestionJobRecord(request=IngestDocumentsRequest(), status="queued")
    await store.create(record)

    monkeypatch.setattr("inference.worker.handlers.ingestion_handler.get_ingestion_job_store", lambda: store)
    monkeypatch.setattr("inference.worker.handlers.ingestion_handler.get_ingestion_service", lambda: service)
    monkeypatch.setattr("inference.worker.handlers.ingestion_handler.get_ingestion_job_result_store", lambda: results)

    processed = await handle_ingestion_jobs(worker_id="worker-1", heartbeat_interval_s=1)
    updated = await store.get(record.job_id)

    assert processed is True
    assert updated.status == "completed"
    assert updated.result_object_key is not None
    assert store.closed is True


async def test_handle_ingestion_jobs_marks_failed_when_service_raises(monkeypatch):
    store = InMemoryIngestionJobStore()
    results = InMemoryJobResultStore()
    service = StaticIngestionService(error=RuntimeError("boom"))
    record = IngestionJobRecord(request=IngestDocumentsRequest(), status="queued")
    await store.create(record)

    monkeypatch.setattr("inference.worker.handlers.ingestion_handler.get_ingestion_job_store", lambda: store)
    monkeypatch.setattr("inference.worker.handlers.ingestion_handler.get_ingestion_service", lambda: service)
    monkeypatch.setattr("inference.worker.handlers.ingestion_handler.get_ingestion_job_result_store", lambda: results)

    processed = await handle_ingestion_jobs(worker_id="worker-1", heartbeat_interval_s=1)
    updated = await store.get(record.job_id)

    assert processed is True
    assert updated.status == "failed"
    assert "RuntimeError: boom" in updated.error
    assert updated.result_object_key is not None
