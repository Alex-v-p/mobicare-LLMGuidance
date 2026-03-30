from __future__ import annotations

import asyncio

import pytest

from inference.worker.heartbeat import cancel_heartbeat, heartbeat_loop, with_heartbeat


class HeartbeatStore:
    def __init__(self, responses: list[bool]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def heartbeat(self, job_id: str, worker_id: str) -> bool:
        self.calls.append((job_id, worker_id))
        return self.responses.pop(0) if self.responses else True


@pytest.mark.asyncio
async def test_heartbeat_loop_stops_when_store_returns_false():
    store = HeartbeatStore([True, False])
    task = asyncio.create_task(heartbeat_loop(store, "job-1", "worker-1", 0))
    await task

    assert store.calls == [("job-1", "worker-1"), ("job-1", "worker-1")]


@pytest.mark.asyncio
async def test_with_heartbeat_cancels_background_task_after_operation():
    store = HeartbeatStore([True, True, True])
    ran = False

    async def operation():
        nonlocal ran
        ran = True
        await asyncio.sleep(0.01)

    await with_heartbeat(store=store, job_id="job-1", worker_id="worker-1", heartbeat_interval_s=0, operation=operation)

    assert ran is True
    assert len(store.calls) >= 1


@pytest.mark.asyncio
async def test_cancel_heartbeat_swallows_cancelled_error():
    task = asyncio.create_task(asyncio.sleep(10))

    await cancel_heartbeat(task)

    assert task.cancelled() is True
