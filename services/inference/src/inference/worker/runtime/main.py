from __future__ import annotations

import asyncio
import os
import traceback

from inference.jobstore import FileJobStore
from inference.pipeline.generate_guidance import GuidancePipeline
from shared.contracts.inference import utc_now_iso


async def run_worker_loop() -> None:
    poll_interval_s = float(os.getenv("WORKER_POLL_INTERVAL_S", "1.0"))
    store = FileJobStore()
    pipeline = GuidancePipeline()

    while True:
        record = store.claim_next()
        if record is None:
            await asyncio.sleep(poll_interval_s)
            continue

        try:
            result = await pipeline.run(record.request)
            record.status = "completed"
            record.result = result
            record.error = None
            record.completed_at = utc_now_iso()
        except Exception as exc:  # pragma: no cover - okay for prototype worker runtime
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            record.completed_at = utc_now_iso()
            metadata = {
                "worker_error_traceback": traceback.format_exc(),
            }
            if record.result is None:
                # keep failure metadata accessible in status responses
                record.request.options.max_tokens = record.request.options.max_tokens
            # stash traceback in a minimal result-like metadata container if desired later
        finally:
            store.update(record)


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
