from __future__ import annotations

import pytest

from inference.worker.executor import execute_next_job
from shared.contracts.inference import InferenceRequest, InferenceResponse, JobRecord
from tests.support.fakes import InMemoryGuidanceJobStore, InMemoryJobResultStore


class StaticRunner:
    def __init__(self, response: InferenceResponse | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response.model_copy(update={"request_id": request.request_id}, deep=True)


class NonMutatingCompletionStore(InMemoryGuidanceJobStore):
    async def mark_completed(
        self,
        record: JobRecord,
        *,
        result: InferenceResponse,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> JobRecord:
        updated = record.model_copy(
            update={
                "status": "completed",
                "result": result,
                "error": None,
                "completed_at": completed_at,
                "result_object_key": result_object_key,
                "lease_expires_at": None,
            },
            deep=True,
        )
        self.records[record.job_id] = updated.model_copy(deep=True)
        return updated

    async def mark_failed(
        self,
        record: JobRecord,
        *,
        error: str,
        completed_at: str,
        result_object_key: str | None = None,
    ) -> JobRecord:
        updated = record.model_copy(
            update={
                "status": "failed",
                "result": None,
                "error": error,
                "completed_at": completed_at,
                "result_object_key": result_object_key,
                "lease_expires_at": None,
            },
            deep=True,
        )
        self.records[record.job_id] = updated.model_copy(deep=True)
        return updated


@pytest.mark.asyncio
async def test_execute_next_job_post_process_does_not_restore_running_state_after_completion():
    store = NonMutatingCompletionStore()
    result_store = InMemoryJobResultStore()
    record = JobRecord(request_id="req-3", status="queued", request=InferenceRequest(request_id="req-3"))
    response = InferenceResponse(request_id="req-3", status="ok", model="test", answer="done")
    await store.create(record)

    async def post_process(job_record: JobRecord) -> None:
        job_record.callback_attempts = 1
        job_record.callback_last_status = "200"

    handled = await execute_next_job(
        store=store,
        result_store=result_store,
        worker_id="worker-3",
        heartbeat_interval_s=1,
        run_request=StaticRunner(response=response).run,
        utc_now_iso=lambda: "2026-03-16T10:30:00+00:00",
        run_with_heartbeat=lambda **kwargs: kwargs["operation"](),
        post_process=post_process,
        job_kind="guidance",
    )

    assert handled is True
    updated = await store.get(record.job_id)
    assert updated is not None
    assert updated.status == "completed"
    assert updated.lease_expires_at is None
    assert updated.callback_attempts == 1
    assert updated.callback_last_status == "200"


@pytest.mark.asyncio
async def test_execute_next_job_completes_and_runs_post_process(monkeypatch):
    store = InMemoryGuidanceJobStore()
    result_store = InMemoryJobResultStore()
    record = JobRecord(request_id="req-1", status="queued", request=InferenceRequest(request_id="req-1"))
    response = InferenceResponse(request_id="req-1", status="ok", model="test", answer="done")
    await store.create(record)

    async def post_process(job_record: JobRecord) -> None:
        job_record.callback_attempts = 3
        job_record.callback_last_status = "200"

    handled = await execute_next_job(
        store=store,
        result_store=result_store,
        worker_id="worker-1",
        heartbeat_interval_s=1,
        run_request=StaticRunner(response=response).run,
        utc_now_iso=lambda: "2026-03-16T10:20:00+00:00",
        run_with_heartbeat=lambda **kwargs: kwargs["operation"](),
        post_process=post_process,
        job_kind="guidance",
    )

    assert handled is True
    updated = await store.get(record.job_id)
    assert updated is not None
    assert updated.status == "completed"
    assert updated.result is not None
    assert updated.result.answer == "done"
    assert updated.callback_attempts == 3
    assert updated.callback_last_status == "200"
    assert updated.result_object_key in result_store.records
    assert store.closed is True


@pytest.mark.asyncio
async def test_execute_next_job_marks_failure_and_still_runs_post_process():
    store = InMemoryGuidanceJobStore()
    result_store = InMemoryJobResultStore()
    record = JobRecord(request_id="req-2", status="queued", request=InferenceRequest(request_id="req-2"))
    await store.create(record)

    async def post_process(job_record: JobRecord) -> None:
        job_record.callback_last_error = "callback skipped"

    handled = await execute_next_job(
        store=store,
        result_store=result_store,
        worker_id="worker-2",
        heartbeat_interval_s=1,
        run_request=StaticRunner(error=RuntimeError("boom")).run,
        utc_now_iso=lambda: "2026-03-16T10:25:00+00:00",
        run_with_heartbeat=lambda **kwargs: kwargs["operation"](),
        post_process=post_process,
        job_kind="guidance",
    )

    assert handled is True
    updated = await store.get(record.job_id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.error == "RuntimeError: boom"
    assert updated.callback_last_error == "callback skipped"
    assert updated.result_object_key in result_store.records
    assert store.closed is True


@pytest.mark.asyncio
async def test_execute_next_job_post_process_does_not_restore_running_state_after_failure():
    store = NonMutatingCompletionStore()
    result_store = InMemoryJobResultStore()
    record = JobRecord(request_id="req-4", status="queued", request=InferenceRequest(request_id="req-4"))
    await store.create(record)

    async def post_process(job_record: JobRecord) -> None:
        job_record.callback_last_error = "callback skipped"

    handled = await execute_next_job(
        store=store,
        result_store=result_store,
        worker_id="worker-4",
        heartbeat_interval_s=1,
        run_request=StaticRunner(error=RuntimeError("boom")).run,
        utc_now_iso=lambda: "2026-03-16T10:35:00+00:00",
        run_with_heartbeat=lambda **kwargs: kwargs["operation"](),
        post_process=post_process,
        job_kind="guidance",
    )

    assert handled is True
    updated = await store.get(record.job_id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.lease_expires_at is None
    assert updated.callback_last_error == "callback skipped"
