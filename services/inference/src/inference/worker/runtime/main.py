from __future__ import annotations

import asyncio
import traceback

from inference.callbacks.notifier import CallbackNotifier
from inference.jobstore.redis_store import RedisJobStore
from inference.pipeline.generate_guidance import GuidancePipeline
from inference.storage.minio_results import MinioResultStore
from shared.contracts.inference import utc_now_iso


async def run_worker_loop() -> None:
    store = RedisJobStore()
    pipeline = GuidancePipeline()
    results = MinioResultStore()
    notifier = CallbackNotifier()

    try:
        while True:
            record = await store.claim_next(timeout_s=5)
            if record is None:
                continue
            if record.status != "running":
                continue

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
                callback_status, callback_error, callback_attempts = await notifier.notify(record)
                record.callback_last_status = callback_status
                record.callback_last_error = callback_error
                record.callback_attempts = callback_attempts
                await store.update(record)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
