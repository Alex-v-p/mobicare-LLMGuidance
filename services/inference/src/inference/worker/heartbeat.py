from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from inference.queue import ClaimableJobStore, RecordT


async def heartbeat_loop(
    store: ClaimableJobStore[RecordT],
    job_id: str,
    worker_id: str,
    interval_s: int,
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        keep_going = await store.heartbeat(job_id, worker_id)
        if not keep_going:
            return


async def cancel_heartbeat(task: asyncio.Task[None]) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def with_heartbeat(
    *,
    store: ClaimableJobStore[RecordT],
    job_id: str,
    worker_id: str,
    heartbeat_interval_s: int,
    operation: Callable[[], Awaitable[None]],
) -> None:
    heartbeat_task = asyncio.create_task(
        heartbeat_loop(store=store, job_id=job_id, worker_id=worker_id, interval_s=heartbeat_interval_s)
    )
    try:
        await operation()
    finally:
        await cancel_heartbeat(heartbeat_task)
