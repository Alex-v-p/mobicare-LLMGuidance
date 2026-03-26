from __future__ import annotations

import traceback

from inference.worker.runtime.dependencies import (
    get_ingestion_job_result_store,
    get_ingestion_job_store,
    get_ingestion_service,
)
from inference.worker.handlers.base import with_heartbeat
from shared.contracts.ingestion import utc_now_iso


async def handle_ingestion_jobs(worker_id: str, heartbeat_interval_s: int) -> bool:
    store = get_ingestion_job_store()
    service = get_ingestion_service()
    results = get_ingestion_job_result_store()
    try:
        record = await store.claim_next(worker_id=worker_id, timeout_s=1)
        if record is None or record.status != "running":
            return False

        async def process() -> None:
            try:
                result = await service.ingest(record.request)
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

        await with_heartbeat(
            store=store,
            job_id=record.job_id,
            worker_id=worker_id,
            heartbeat_interval_s=heartbeat_interval_s,
            operation=process,
        )

        record.worker_id = worker_id
        record.lease_expires_at = None
        await store.update(record)
        return True
    finally:
        await store.close()
