from __future__ import annotations

import httpx
import pytest

from api.main import create_app as create_api_app
from inference.infrastructure.http.dependencies import get_guidance_job_service
from inference.infrastructure.http.main import create_app as create_inference_app
from inference.application.services.guidance_service import GuidanceJobService
from inference.worker.handlers import guidance_handler
from shared.contracts.inference import ApiGuidanceJobStatus
from tests.support.fakes import DummyAsyncClientContext, InMemoryGuidanceJobStore, InMemoryJobResultStore, RecordingNotifier, StaticGuidancePipeline

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_async_guidance_job_flow_end_to_end(guidance_request, inference_response, monkeypatch):
    store = InMemoryGuidanceJobStore()
    result_store = InMemoryJobResultStore()
    notifier = RecordingNotifier()

    inference_app = create_inference_app()
    inference_app.dependency_overrides[get_guidance_job_service] = lambda: GuidanceJobService(
        store_factory=lambda: store,
        result_store=result_store,
    )

    class AlwaysReadyRetrievalState:
        async def is_guidance_ready(self) -> bool:
            return True

    monkeypatch.setattr(guidance_handler, "get_guidance_job_store", lambda: store)
    monkeypatch.setattr(guidance_handler, "get_guidance_pipeline", lambda: StaticGuidancePipeline(inference_response))
    monkeypatch.setattr(guidance_handler, "get_guidance_job_result_store", lambda: result_store)
    monkeypatch.setattr(guidance_handler, "get_retrieval_state_controller", lambda: AlwaysReadyRetrievalState())
    monkeypatch.setattr(guidance_handler, "CallbackNotifier", lambda: notifier)
    monkeypatch.setattr(guidance_handler, "with_heartbeat", lambda **kwargs: kwargs["operation"]())
    monkeypatch.setattr(guidance_handler, "utc_now_iso", lambda: "2026-03-16T10:30:00+00:00")
    monkeypatch.setattr(
        "api.infrastructure.clients.inference_client.create_async_client",
        lambda timeout_s: DummyAsyncClientContext(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=inference_app),
                base_url="http://inference.test",
                timeout=timeout_s,
            )
        ),
    )

    api_app = create_api_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://api.test") as client:
        create_response = await client.post("/guidance/jobs", json=guidance_request.model_dump(mode="json"))
        assert create_response.status_code == 200
        job_id = create_response.json()["job_id"]

        handled = await guidance_handler.handle_guidance_jobs(worker_id="worker-e2e", heartbeat_interval_s=5)
        assert handled is True

        status_response = await client.get(f"/guidance/jobs/{job_id}")
