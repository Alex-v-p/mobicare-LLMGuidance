from __future__ import annotations

import asyncio
import traceback
from uuid import uuid4

from inference.callbacks.notifier import CallbackNotifier
from inference.http.dependencies import (
    get_guidance_job_result_store,
    get_guidance_job_store,
    get_guidance_pipeline,
    get_ingestion_job_result_store,
    get_ingestion_job_store,
    get_ingestion_service,
)
from shared.config import get_settings
from shared.contracts.ingestion import utc_now_iso as ingestion_utc_now_iso
from shared.contracts.inference import utc_now_iso


async def _heartbeat_loop(store, job_id: str, worker_id: str, interval_s: int) -> None:
    while True:
        await asyncio.sleep(interval_s)
        keep_going = await store.heartbeat(job_id, worker_id)
        if not keep_going:
            return


async def _handle_guidance_jobs(worker_id: str, heartbeat_interval_s: int) -> bool:
    store = get_guidance_job_store()
    pipeline = get_guidance_pipeline()
    results = get_guidance_job_result_store()
    notifier = CallbackNotifier()
    try:
        record = await store.claim_next(worker_id=worker_id, timeout_s=1)
        if record is None or record.status != "running":
            return False

        heartbeat_task = asyncio.create_task(_heartbeat_loop(store, record.job_id, worker_id, heartbeat_interval_s))
        try:
            result = await pipeline.run(record.request)
            completed_at = utc_now_iso()
            record.result_object_key = results.put_job_result(
                record.model_copy(
                    update={
                        "status": "completed",
                        "result": result,
                        "error": None,
                        "completed_at": completed_at,
                        "lease_expires_at": None,
                    }
                )
            )
            await store.mark_completed(
                record,
                result=result,
                completed_at=completed_at,
                result_object_key=record.result_object_key,
            )
        except Exception as exc:  # pragma: no cover
            failed_at = utc_now_iso()
            record.result_object_key = results.put_job_result(
                record.model_copy(
                    update={
                        "status": "failed",
                        "result": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "completed_at": failed_at,
                        "lease_expires_at": None,
                    }
                )
            )
            await store.mark_failed(
                record,
                error=f"{type(exc).__name__}: {exc}",
                completed_at=failed_at,
                result_object_key=record.result_object_key,
            )
            traceback.print_exc()
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

            record.worker_id = worker_id
            record.lease_expires_at = None

            callback_status, callback_error, callback_attempts = await notifier.notify(
                record.request.options.callback_url,
                record.request.options.callback_headers,
                record,
            )
            record.callback_last_status = callback_status
            record.callback_last_error = callback_error
            record.callback_attempts = callback_attempts

            await store.update(record)
        return True
    finally:
        await store.close()


async def _handle_ingestion_jobs(worker_id: str, heartbeat_interval_s: int) -> bool:
    store = get_ingestion_job_store()
    service = get_ingestion_service()
    results = get_ingestion_job_result_store()
    try:
        record = await store.claim_next(worker_id=worker_id, timeout_s=1)
        if record is None or record.status != "running":
            return False

        heartbeat_task = asyncio.create_task(_heartbeat_loop(store, record.job_id, worker_id, heartbeat_interval_s))
        try:
            result = await service.ingest(record.request)
            completed_at = ingestion_utc_now_iso()
            record.result_object_key = results.put_job_result(
                record.model_copy(
                    update={
                        "status": "completed",
                        "result": result,
                        "error": None,
                        "completed_at": completed_at,
                        "lease_expires_at": None,
                    }
                )
            )
            await store.mark_completed(
                record,
                result=result,
                completed_at=completed_at,
                result_object_key=record.result_object_key,
            )
        except Exception as exc:  # pragma: no cover
            failed_at = ingestion_utc_now_iso()
            record.result_object_key = results.put_job_result(
                record.model_copy(
                    update={
                        "status": "failed",
                        "result": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "completed_at": failed_at,
                        "lease_expires_at": None,
                    }
                )
            )
            await store.mark_failed(
                record,
                error=f"{type(exc).__name__}: {exc}",
                completed_at=failed_at,
                result_object_key=record.result_object_key,
            )
            traceback.print_exc()
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            record.worker_id = worker_id
            record.lease_expires_at = None
            await store.update(record)
        return True
    finally:
        await store.close()


async def run_worker_loop() -> None:
    settings = get_settings()
    worker_id = settings.worker_id or f"worker-{uuid4()}"
    heartbeat_interval_s = settings.job_heartbeat_interval_seconds

    while True:
        handled = await _handle_guidance_jobs(worker_id=worker_id, heartbeat_interval_s=heartbeat_interval_s)
        if handled:
            continue
        handled = await _handle_ingestion_jobs(worker_id=worker_id, heartbeat_interval_s=heartbeat_interval_s)
        if handled:
            continue
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
