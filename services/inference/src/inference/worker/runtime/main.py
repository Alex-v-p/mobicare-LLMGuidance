from __future__ import annotations

import asyncio
import os
import traceback
from uuid import uuid4

from inference.callbacks.notifier import CallbackNotifier
from inference.jobstore.redis_store import RedisJobStore
from inference.pipeline.generate_guidance import GuidancePipeline
from inference.storage.minio_results import MinioResultStore
from shared.contracts.inference import utc_now_iso


async def _heartbeat_loop(store: RedisJobStore, job_id: str, worker_id: str, interval_s: int) -> None:
    while True:
        await asyncio.sleep(interval_s)
        keep_going = await store.heartbeat(job_id, worker_id)
        if not keep_going:
            return


async def run_worker_loop() -> None:
    store = RedisJobStore()
    pipeline = GuidancePipeline()
    results = MinioResultStore()
    notifier = CallbackNotifier()
    worker_id = os.getenv("WORKER_ID", f"worker-{uuid4()}")
    heartbeat_interval_s = int(os.getenv("JOB_HEARTBEAT_INTERVAL_SECONDS", "20"))

    try:
        while True:
            record = await store.claim_next(worker_id=worker_id, timeout_s=5)
            if record is None:
                continue
            if record.status != "running":
                continue

            heartbeat_task = asyncio.create_task(_heartbeat_loop(store, record.job_id, worker_id, heartbeat_interval_s))
            try:
                result = await pipeline.run(record.request)
                record.status = "completed"
                record.result = result
                record.error = None
                record.completed_at = utc_now_iso()
                record.result_object_key = results.put_job_result(record)
            except Exception as exc:  # pragma: no cover
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
                record.completed_at = utc_now_iso()
                record.result = None
                record.result_object_key = results.put_job_result(record)
                traceback.print_exc()
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                record.worker_id = worker_id
                record.lease_expires_at = None
                callback_status, callback_error, callback_attempts = await notifier.notify(record)
                record.callback_last_status = callback_status
                record.callback_last_error = callback_error
                record.callback_attempts = callback_attempts
                await store.update(record)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
