from __future__ import annotations

import asyncio
from uuid import uuid4

from inference.worker.runtime.dependencies import get_document_store, get_guidance_job_result_store, get_ingestion_job_result_store, get_worker_settings
from inference.worker.handlers import handle_guidance_jobs, handle_ingestion_jobs
from shared.bootstrap.minio import bootstrap_minio_resources_on_startup


async def run_worker_loop() -> None:
    settings = get_worker_settings()
    bootstrap_minio_resources_on_startup(
        settings=settings,
        client=get_document_store().client,
        service="inference-worker",
    )
    get_guidance_job_result_store().ensure_bucket()
    get_ingestion_job_result_store().ensure_bucket()
    worker_id = settings.worker_id or f"worker-{uuid4()}"
    heartbeat_interval_s = settings.job_heartbeat_interval_seconds

    while True:
        handled = await handle_guidance_jobs(worker_id=worker_id, heartbeat_interval_s=heartbeat_interval_s)
        if handled:
            continue
        handled = await handle_ingestion_jobs(worker_id=worker_id, heartbeat_interval_s=heartbeat_interval_s)
        if handled:
            continue
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
