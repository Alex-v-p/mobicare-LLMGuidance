from __future__ import annotations

import asyncio

import pytest

from inference.worker.runtime import main as runtime_main


@pytest.mark.asyncio
async def test_run_worker_loop_prioritizes_ingestion_before_guidance(monkeypatch):
    calls: list[str] = []

    class DummySettings:
        worker_id = "worker-test"
        job_heartbeat_interval_seconds = 5

    class DummyRetrievalState:
        async def refresh_from_vector_store(self):
            return None

        async def close(self):
            return None

    shutdown_event = asyncio.Event()

    monkeypatch.setattr(runtime_main, "get_worker_settings", lambda: DummySettings())
    monkeypatch.setattr(runtime_main, "get_retrieval_state_controller", lambda: DummyRetrievalState())
    monkeypatch.setattr(runtime_main, "_install_signal_handlers", lambda _stop_event: None)

    async def fake_handle_ingestion_jobs(*, worker_id: str, heartbeat_interval_s: int) -> bool:
        calls.append("ingestion")
        return False

    async def fake_handle_guidance_jobs(*, worker_id: str, heartbeat_interval_s: int) -> bool:
        calls.append("guidance")
        return False

    async def stop_after_first_idle(*, stop_event: asyncio.Event, poll_interval_s: float = 1.0) -> None:
        shutdown_event.set()

    monkeypatch.setattr(runtime_main, "handle_ingestion_jobs", fake_handle_ingestion_jobs)
    monkeypatch.setattr(runtime_main, "handle_guidance_jobs", fake_handle_guidance_jobs)
    monkeypatch.setattr(runtime_main, "_sleep_until_next_poll", stop_after_first_idle)

    await runtime_main.run_worker_loop(stop_event=shutdown_event)

    assert calls == ["ingestion", "guidance"]


@pytest.mark.asyncio
async def test_run_worker_loop_stops_after_sigterm_request(monkeypatch):
    class DummySettings:
        worker_id = "worker-test"
        job_heartbeat_interval_seconds = 5

    class DummyRetrievalState:
        def __init__(self) -> None:
            self.closed = False

        async def refresh_from_vector_store(self):
            return None

        async def close(self):
            self.closed = True

    retrieval_state = DummyRetrievalState()
    shutdown_event = asyncio.Event()

    monkeypatch.setattr(runtime_main, "get_worker_settings", lambda: DummySettings())
    monkeypatch.setattr(runtime_main, "get_retrieval_state_controller", lambda: retrieval_state)

    def install_handlers(stop_event: asyncio.Event) -> None:
        stop_event.set()

    async def fake_handle_ingestion_jobs(*, worker_id: str, heartbeat_interval_s: int) -> bool:
        return False

    async def fake_handle_guidance_jobs(*, worker_id: str, heartbeat_interval_s: int) -> bool:
        raise AssertionError("guidance should not run after shutdown was requested")

    monkeypatch.setattr(runtime_main, "_install_signal_handlers", install_handlers)
    monkeypatch.setattr(runtime_main, "handle_ingestion_jobs", fake_handle_ingestion_jobs)
    monkeypatch.setattr(runtime_main, "handle_guidance_jobs", fake_handle_guidance_jobs)

    await runtime_main.run_worker_loop(stop_event=shutdown_event)

    assert retrieval_state.closed is True
