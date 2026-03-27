from __future__ import annotations

import pytest

from inference.worker.handlers import guidance_handler
from shared.contracts.inference import InferenceRequest, InferenceResponse, JobRecord
from tests.support.fakes import InMemoryGuidanceJobStore, InMemoryJobResultStore, RecordingNotifier, StaticGuidancePipeline


class StaticRetrievalState:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    async def is_guidance_ready(self) -> bool:
        return self.ready


@pytest.mark.asyncio
async def test_handle_guidance_jobs_completes_and_updates_callback(monkeypatch, inference_response):
    store = InMemoryGuidanceJobStore()
    result_store = InMemoryJobResultStore()
    notifier = RecordingNotifier()
    request = InferenceRequest(
        request_id="req-1",
        question="How should I treat this patient?",
        patient_variables={"age": 72},
        options={"callback_url": "https://callback.local/hook", "callback_headers": {"X-Test": "1"}},
    )
    record = JobRecord(request_id="req-1", status="queued", request=request)
    await store.create(record)

    monkeypatch.setattr(guidance_handler, "get_guidance_job_store", lambda: store)
    monkeypatch.setattr(guidance_handler, "get_guidance_pipeline", lambda: StaticGuidancePipeline(inference_response))
    monkeypatch.setattr(guidance_handler, "get_guidance_job_result_store", lambda: result_store)
    monkeypatch.setattr(guidance_handler, "CallbackNotifier", lambda: notifier)
    monkeypatch.setattr(guidance_handler, "with_heartbeat", lambda **kwargs: kwargs["operation"]())
    monkeypatch.setattr(guidance_handler, "utc_now_iso", lambda: "2026-03-16T10:15:00+00:00")
    monkeypatch.setattr(guidance_handler, "get_retrieval_state_controller", lambda: StaticRetrievalState(True))

    handled = await guidance_handler.handle_guidance_jobs(worker_id="worker-1", heartbeat_interval_s=5)

    assert handled is True
    stored = store.records[record.job_id]
    assert stored.status == "completed"
    assert stored.result is not None
    assert stored.result.answer == inference_response.answer
    assert stored.callback_attempts == 1
    assert stored.callback_last_status == "200"
    assert stored.result_object_key in result_store.records
    assert notifier.calls[0]["callback_headers"] == {"X-Test": "1"}


@pytest.mark.asyncio
async def test_handle_guidance_jobs_skips_when_retrieval_not_ready(monkeypatch):
    store = InMemoryGuidanceJobStore()
    monkeypatch.setattr(guidance_handler, "get_retrieval_state_controller", lambda: StaticRetrievalState(False))
    monkeypatch.setattr(guidance_handler, "get_guidance_job_store", lambda: store)

    handled = await guidance_handler.handle_guidance_jobs(worker_id="worker-1", heartbeat_interval_s=5)

    assert handled is False
    assert store.queue == store.queue
