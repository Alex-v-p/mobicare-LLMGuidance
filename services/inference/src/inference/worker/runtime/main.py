from __future__ import annotations

import asyncio
import signal
from contextlib import suppress
from uuid import uuid4

from inference.worker.runtime.dependencies import (
    get_document_store,
    get_guidance_job_result_store,
    get_ingestion_job_result_store,
    get_retrieval_state_controller,
    get_worker_settings,
)
from inference.worker.handlers import handle_guidance_jobs, handle_ingestion_jobs
from shared.observability import get_logger

logger = get_logger(__name__, service="inference-worker")


async def _sleep_until_next_poll(*, stop_event: asyncio.Event, poll_interval_s: float = 1.0) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_s)
    except asyncio.TimeoutError:
        return


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def _request_shutdown(signame: str) -> None:
        if stop_event.is_set():
            return
        logger.info(
            "worker_shutdown_requested",
            extra={
                "event": "worker_shutdown_requested",
                "signal": signame,
            },
        )
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_shutdown, sig.name)


async def run_worker_loop(stop_event: asyncio.Event | None = None) -> None:
    settings = get_worker_settings()
    retrieval_state = get_retrieval_state_controller()
    await retrieval_state.refresh_from_vector_store()

    worker_id = settings.worker_id or f"worker-{uuid4()}"
    heartbeat_interval_s = settings.job_heartbeat_interval_seconds
    shutdown_event = stop_event or asyncio.Event()

    _install_signal_handlers(shutdown_event)

    try:
        while not shutdown_event.is_set():
            handled = await handle_ingestion_jobs(worker_id=worker_id, heartbeat_interval_s=heartbeat_interval_s)
            if handled:
                continue
            if shutdown_event.is_set():
                break

            handled = await handle_guidance_jobs(worker_id=worker_id, heartbeat_interval_s=heartbeat_interval_s)
            if handled:
                continue
            if shutdown_event.is_set():
                break

            await _sleep_until_next_poll(stop_event=shutdown_event)
    finally:
        await retrieval_state.close()
        logger.info(
            "worker_shutdown_complete",
            extra={
                "event": "worker_shutdown_complete",
                "worker_id": worker_id,
            },
        )


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
