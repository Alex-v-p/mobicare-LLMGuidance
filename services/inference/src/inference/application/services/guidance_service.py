from __future__ import annotations

from collections.abc import Callable

from inference.control.retrieval_state import RetrievalStateController
from inference.infrastructure.http.exceptions import NotFoundError
from inference.jobstore.base import ReadWriteJobStore, managed_store
from inference.application.pipelines.guidance_pipeline import GuidancePipeline
from inference.storage.base_minio_job_results import JobResultStore
from shared.contracts.inference import InferenceRequest, InferenceResponse, JobRecord
from shared.observability import get_logger


logger = get_logger(__name__, service="inference")


class GuidanceRequestService:
    def __init__(self, pipeline: GuidancePipeline, retrieval_state: RetrievalStateController | None = None) -> None:
        self._pipeline = pipeline
        self._retrieval_state = retrieval_state

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        if self._retrieval_state is not None:
            await self._retrieval_state.ensure_guidance_ready()
        return await self._pipeline.run(request)


class GuidanceJobService:
    def __init__(
        self,
        *,
        store_factory: Callable[[], ReadWriteJobStore[JobRecord]],
        result_store: JobResultStore[JobRecord],
        retrieval_state: RetrievalStateController | None = None,
    ) -> None:
        self._store_factory = store_factory
        self._result_store = result_store
        self._retrieval_state = retrieval_state

    async def create(self, request: InferenceRequest) -> JobRecord:
        if self._retrieval_state is not None:
            await self._retrieval_state.ensure_guidance_ready()
        record = JobRecord(request_id=request.request_id, status="queued", request=request)
        async with managed_store(self._store_factory) as store:
            await store.create(record)
            return record

    async def get(self, job_id: str) -> JobRecord:
        async with managed_store(self._store_factory) as store:
            record = await store.get(job_id)

        if record is None:
            raise NotFoundError(f"Job {job_id} was not found")

        if record.result is None and record.result_object_key:
            try:
                record = self._result_store.get_job_result(record.result_object_key)
            except Exception as exc:
                logger.warning(
                    "guidance_job_archived_result_load_failed",
                    extra={
                        "event": "guidance_job_archived_result_load_failed",
                        "error_code": "JOB_RESULT_LOAD_FAILED",
                    },
                    exc_info=exc,
                )
        return record
