from __future__ import annotations

import pytest

from inference.worker.runtime import main as runtime_main


@pytest.mark.asyncio
async def test_run_worker_loop_prioritizes_ingestion_before_guidance(monkeypatch):
    calls: list[str] = []

    class DummySettings:
        worker_id = "worker-test"
        job_heartbeat_interval_seconds = 5

    monkeypatch.setattr(runtime_main, "get_worker_settings", lambda: DummySettings())
    monkeypatch.setattr(runtime_main, "bootstrap_minio_resources_on_startup", lambda **kwargs: None)

    class DummyStore:
        client = object()

    class DummyResultStore:
        def ensure_bucket(self) -> None:
            return None

    monkeypatch.setattr(runtime_main, "get_document_store", lambda: DummyStore())
    class DummyRetrievalState:
        async def refresh_from_vector_store(self):
            return None

    monkeypatch.setattr(runtime_main, "get_guidance_job_result_store", lambda: DummyResultStore())
    monkeypatch.setattr(runtime_main, "get_ingestion_job_result_store", lambda: DummyResultStore())
    monkeypatch.setattr(runtime_main, "get_retrieval_state_controller", lambda: DummyRetrievalState())

    async def fake_handle_ingestion_jobs(*, worker_id: str, heartbeat_interval_s: int) -> bool:
        calls.append("ingestion")
        return False

    async def fake_handle_guidance_jobs(*, worker_id: str, heartbeat_interval_s: int) -> bool:
        calls.append("guidance")
        return False

    async def stop_after_first_idle(_: int) -> None:
        raise RuntimeError("stop loop")

    monkeypatch.setattr(runtime_main, "handle_ingestion_jobs", fake_handle_ingestion_jobs)
    monkeypatch.setattr(runtime_main, "handle_guidance_jobs", fake_handle_guidance_jobs)
    monkeypatch.setattr(runtime_main.asyncio, "sleep", stop_after_first_idle)

    with pytest.raises(RuntimeError, match="stop loop"):
        await runtime_main.run_worker_loop()

    assert calls == ["ingestion", "guidance"]
