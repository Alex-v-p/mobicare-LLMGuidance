from __future__ import annotations

from collections.abc import Callable

from inference.infrastructure.http.exceptions import NotFoundError
from inference.jobstore.base import ReadWriteJobStore, managed_store
from inference.indexing.ingestion_service import IngestionService
from inference.storage.qdrant_store import QdrantVectorStore
from inference.storage.base_minio_job_results import JobResultStore
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionCollectionDeleteResponse, IngestionJobRecord, IngestionResponse
from shared.observability import get_logger


logger = get_logger(__name__, service="inference")


class IngestionRequestService:
    def __init__(self, ingestion_service: IngestionService, vector_store: QdrantVectorStore) -> None:
        self._ingestion_service = ingestion_service
        self._vector_store = vector_store

    async def ingest(self, payload: IngestDocumentsRequest) -> IngestionResponse:
        return await self._ingestion_service.ingest(payload)

    async def delete_collection(self) -> IngestionCollectionDeleteResponse:
        existed = self._vector_store.delete_collection()
        return IngestionCollectionDeleteResponse(collection=self._vector_store.collection_name, existed=existed)


class IngestionJobService:
    def __init__(
        self,
        *,
        store_factory: Callable[[], ReadWriteJobStore[IngestionJobRecord]],
        result_store: JobResultStore[IngestionJobRecord],
    ) -> None:
        self._store_factory = store_factory
        self._result_store = result_store

    async def create(self, payload: IngestDocumentsRequest) -> IngestionJobRecord:
        record = IngestionJobRecord(status="queued", request=payload)
        async with managed_store(self._store_factory) as store:
            await store.create(record)
            return record

    async def get(self, job_id: str) -> IngestionJobRecord:
        async with managed_store(self._store_factory) as store:
            record = await store.get(job_id)

        if record is None:
            raise NotFoundError(f"Ingestion job {job_id} was not found")

        if record.result is None and record.result_object_key:
            try:
                record = self._result_store.get_job_result(record.result_object_key)
            except Exception as exc:
                logger.warning(
                    "ingestion_job_archived_result_load_failed",
                    extra={
                        "event": "ingestion_job_archived_result_load_failed",
                        "error_code": "JOB_RESULT_LOAD_FAILED",
                    },
                    exc_info=exc,
                )
        return record
