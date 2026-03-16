from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from shared.contracts.inference import InferenceRequest, InferenceResponse, JobRecord
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionJobRecord, IngestionResponse


class DummyAsyncClientContext:
    def __init__(self, client):
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, exc_type, exc, tb):
        await self._client.aclose()
        return False


class InMemoryGuidanceJobStore:
    def __init__(self) -> None:
        self.records: dict[str, JobRecord] = {}
        self.queue: deque[str] = deque()
        self.closed = False

    async def create(self, record: JobRecord) -> None:
        if record.job_id in self.records:
            raise FileExistsError(record.job_id)
        self.records[record.job_id] = record.model_copy(deep=True)
        self.queue.append(record.job_id)

    async def get(self, job_id: str) -> JobRecord | None:
        record = self.records.get(job_id)
        return record.model_copy(deep=True) if record is not None else None

    async def update(self, record: JobRecord) -> None:
        self.records[record.job_id] = record.model_copy(deep=True)

    async def close(self) -> None:
        self.closed = True

    async def claim_next(self, worker_id: str, timeout_s: int = 5) -> JobRecord | None:
        if not self.queue:
            return None
        job_id = self.queue.popleft()
        record = self.records[job_id].model_copy(deep=True)
        if record.status == "queued":
            record.status = "running"
            record.worker_id = worker_id
            record.started_at = record.started_at or "2026-03-16T10:00:00+00:00"
            record.lease_expires_at = "2026-03-16T10:05:00+00:00"
            self.records[job_id] = record.model_copy(deep=True)
        return record

    async def heartbeat(self, job_id: str, worker_id: str) -> bool:
        record = self.records.get(job_id)
        return bool(record and record.status == "running" and record.worker_id == worker_id)

    async def mark_completed(
        self,
        record: JobRecord,
        *,
        result: InferenceResponse,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> JobRecord:
        record.status = "completed"
        record.result = result
        record.error = None
        record.completed_at = completed_at
        record.result_object_key = result_object_key
        record.lease_expires_at = None
        updated = record.model_copy(deep=True)
        self.records[record.job_id] = updated
        return updated

    async def mark_failed(
        self,
        record: JobRecord,
        *,
        error: str,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> JobRecord:
        record.status = "failed"
        record.result = None
        record.error = error
        record.completed_at = completed_at
        record.result_object_key = result_object_key
        record.lease_expires_at = None
        updated = record.model_copy(deep=True)
        self.records[record.job_id] = updated
        return updated


class InMemoryIngestionJobStore:
    def __init__(self) -> None:
        self.records: dict[str, IngestionJobRecord] = {}
        self.queue: deque[str] = deque()
        self.closed = False

    async def create(self, record: IngestionJobRecord) -> None:
        if record.job_id in self.records:
            raise FileExistsError(record.job_id)
        self.records[record.job_id] = record.model_copy(deep=True)
        self.queue.append(record.job_id)

    async def get(self, job_id: str) -> IngestionJobRecord | None:
        record = self.records.get(job_id)
        return record.model_copy(deep=True) if record is not None else None

    async def update(self, record: IngestionJobRecord) -> None:
        self.records[record.job_id] = record.model_copy(deep=True)

    async def close(self) -> None:
        self.closed = True


class InMemoryJobResultStore:
    def __init__(self) -> None:
        self.records: dict[str, Any] = {}
        self.counter = 0

    def put_job_result(self, record):
        self.counter += 1
        key = f"jobs/result-{self.counter}.json"
        self.records[key] = record.model_copy(deep=True)
        return key

    def get_job_result(self, object_key: str):
        return self.records[object_key].model_copy(deep=True)


@dataclass
class StaticGuidancePipeline:
    response: InferenceResponse

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        return self.response.model_copy(
            update={"request_id": request.request_id, "used_variables": deepcopy(request.patient_variables)},
            deep=True,
        )


@dataclass
class StaticIngestionService:
    response: IngestionResponse

    async def ingest(self, payload: IngestDocumentsRequest) -> IngestionResponse:
        return self.response.model_copy(deep=True)


class RecordingNotifier:
    def __init__(self, result: tuple[str | None, str | None, int] = ("200", None, 1)) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result

    async def notify(self, callback_url, callback_headers, record):
        self.calls.append(
            {
                "callback_url": callback_url,
                "callback_headers": dict(callback_headers or {}),
                "record": record.model_copy(deep=True),
            }
        )
        return self._result
