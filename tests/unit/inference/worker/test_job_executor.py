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
