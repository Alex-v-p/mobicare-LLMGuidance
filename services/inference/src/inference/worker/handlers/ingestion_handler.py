from __future__ import annotations

from inference.worker.executor import execute_next_job
from inference.worker.runtime.dependencies import (
    get_ingestion_job_result_store,
    get_ingestion_job_store,
    get_ingestion_service,
    get_retrieval_state_controller,
)
from inference.worker.heartbeat import with_heartbeat
from shared.contracts.ingestion import IngestionResponse, utc_now_iso


async def _mark_ingesting(record) -> None:
    await get_retrieval_state_controller().mark_ingesting(job_id=record.job_id)


async def _mark_ready(_record, result: IngestionResponse) -> None:
    await get_retrieval_state_controller().mark_ready(
        collection=result.collection,
        embedding_model=result.embedding_model,
    )


async def _mark_failed(_record, error: str) -> None:
    await get_retrieval_state_controller().mark_failed(error)


async def handle_ingestion_jobs(worker_id: str, heartbeat_interval_s: int) -> bool:
    return await execute_next_job(
        store=get_ingestion_job_store(),
        result_store=get_ingestion_job_result_store(),
        worker_id=worker_id,
        heartbeat_interval_s=heartbeat_interval_s,
        run_request=get_ingestion_service().ingest,
        utc_now_iso=utc_now_iso,
        run_with_heartbeat=with_heartbeat,
        before_run=_mark_ingesting,
        after_success=_mark_ready,
        after_failure=_mark_failed,
        job_kind="ingestion",
    )
