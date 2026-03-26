from __future__ import annotations

from inference.worker.executor import execute_next_job
from inference.worker.runtime.dependencies import (
    get_ingestion_job_result_store,
    get_ingestion_job_store,
    get_ingestion_service,
)
from inference.worker.heartbeat import with_heartbeat
from shared.contracts.ingestion import utc_now_iso


async def handle_ingestion_jobs(worker_id: str, heartbeat_interval_s: int) -> bool:
    return await execute_next_job(
        store=get_ingestion_job_store(),
        result_store=get_ingestion_job_result_store(),
        worker_id=worker_id,
        heartbeat_interval_s=heartbeat_interval_s,
        run_request=get_ingestion_service().ingest,
        utc_now_iso=utc_now_iso,
        run_with_heartbeat=with_heartbeat,
        job_kind="ingestion",
    )
