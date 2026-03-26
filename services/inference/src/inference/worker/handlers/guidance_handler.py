from __future__ import annotations

from inference.callbacks.notifier import CallbackNotifier
from inference.worker.executor import execute_next_job
from inference.worker.runtime.dependencies import (
    get_guidance_job_result_store,
    get_guidance_job_store,
    get_guidance_pipeline,
)
from inference.worker.heartbeat import with_heartbeat
from shared.contracts.inference import JobRecord, utc_now_iso


async def _notify_callback(record: JobRecord) -> None:
    notifier = CallbackNotifier()
    callback_status, callback_error, callback_attempts = await notifier.notify(
        record.request.options.callback_url,
        record.request.options.callback_headers,
        record,
    )
    record.callback_last_status = callback_status
    record.callback_last_error = callback_error
    record.callback_attempts = callback_attempts


async def handle_guidance_jobs(worker_id: str, heartbeat_interval_s: int) -> bool:
    return await execute_next_job(
        store=get_guidance_job_store(),
        result_store=get_guidance_job_result_store(),
        worker_id=worker_id,
        heartbeat_interval_s=heartbeat_interval_s,
        run_request=get_guidance_pipeline().run,
        utc_now_iso=utc_now_iso,
        run_with_heartbeat=with_heartbeat,
        post_process=_notify_callback,
        job_kind="guidance",
    )
