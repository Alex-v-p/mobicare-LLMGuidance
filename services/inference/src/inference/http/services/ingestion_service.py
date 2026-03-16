from __future__ import annotations

from collections.abc import Callable

from inference.http.exceptions import NotFoundError
from inference.indexing.ingestion_service import IngestionService
from inference.storage.qdrant_store import QdrantVectorStore
from inference.jobstore.redis_ingestion_job_store import RedisIngestionJobStore
from inference.storage.minio_ingestion_job_results import MinioIngestionJobResultStore
from shared.contracts.ingestion import IngestDocumentsRequest, IngestionCollectionDeleteResponse, IngestionJobRecord, IngestionResponse


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
        store_factory: Callable[[], RedisIngestionJobStore],
        result_store: MinioIngestionJobResultStore,
    ) -> None:
        self._store_factory = store_factory
        self._result_store = result_store

    async def create(self, payload: IngestDocumentsRequest) -> IngestionJobRecord:
        store = self._store_factory()
        record = IngestionJobRecord(status="queued", request=payload)
        try:
            await store.create(record)
            return record
        finally:
            await store.close()

    async def get(self, job_id: str) -> IngestionJobRecord:
        store = self._store_factory()
        try:
            record = await store.get(job_id)
        finally:
            await store.close()

        if record is None:
            raise NotFoundError(f"Ingestion job {job_id} was not found")

        if record.result is None and record.result_object_key:
            try:
                record = self._result_store.get_job_result(record.result_object_key)
            except Exception:
                pass
        return record
