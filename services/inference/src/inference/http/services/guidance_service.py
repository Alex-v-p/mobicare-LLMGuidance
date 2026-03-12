from __future__ import annotations

from collections.abc import Callable

from inference.http.exceptions import NotFoundError
from inference.jobstore.redis_guidance_job_store import RedisGuidanceJobStore
from inference.pipeline.generate_guidance import GuidancePipeline
from inference.storage.minio_guidance_job_results import MinioGuidanceJobResultStore
from shared.contracts.inference import InferenceRequest, InferenceResponse, JobRecord


class GuidanceRequestService:
    def __init__(self, pipeline: GuidancePipeline) -> None:
        self._pipeline = pipeline

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return await self._pipeline.run(request)


class GuidanceJobService:
    def __init__(
        self,
        *,
        store_factory: Callable[[], RedisGuidanceJobStore],
        result_store: MinioGuidanceJobResultStore,
    ) -> None:
        self._store_factory = store_factory
        self._result_store = result_store

    async def create(self, request: InferenceRequest) -> JobRecord:
        store = self._store_factory()
        record = JobRecord(request_id=request.request_id, status="queued", request=request)
        try:
            await store.create(record)
            return record
        finally:
            await store.close()

    async def get(self, job_id: str) -> JobRecord:
        store = self._store_factory()
        try:
            record = await store.get(job_id)
        finally:
            await store.close()

        if record is None:
            raise NotFoundError(f"Job {job_id} was not found")

        if record.result is None and record.result_object_key:
            try:
                record = self._result_store.get_job_result(record.result_object_key)
            except Exception:
                pass
        return record
