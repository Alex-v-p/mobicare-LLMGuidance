from __future__ import annotations

import asyncio

import pytest

from inference.jobstore.redis_guidance_job_store import RedisGuidanceJobStore
from shared.contracts.inference import InferenceRequest, InferenceResponse, JobRecord
from tests.support.docker import managed_container, reserve_tcp_port, wait_for_redis, require_docker


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_redis_guidance_job_store_round_trip():
    require_docker()
    port = reserve_tcp_port()
    with managed_container(image="redis:7.2-alpine", ports={port: 6379}):
        await wait_for_redis("127.0.0.1", port)

        store = RedisGuidanceJobStore(
            redis_url=f"redis://127.0.0.1:{port}/0",
            queue_name="guidance_jobs_test",
            ttl_seconds=120,
            lease_seconds=30,
        )
        record = JobRecord(
            request_id="req-1",
            status="queued",
            request=InferenceRequest(request_id="req-1", question="What now?"),
        )

        for attempt in range(3):
            try:
                await store.create(record)
                break
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(1)

        claimed = await store.claim_next(worker_id="worker-1", timeout_s=1)
        assert claimed is not None
        assert claimed.status == "running"
        assert await store.heartbeat(claimed.job_id, "worker-1") is True

        completed = await store.mark_completed(
            claimed,
            result=InferenceResponse(request_id="req-1", status="ok", model="m", answer="a"),
            completed_at="2026-03-16T10:00:00+00:00",
            result_object_key="jobs/result.json",
        )
        loaded = await store.get(record.job_id)
        await store.close()

    assert completed.status == "completed"
    assert loaded is not None
    assert loaded.result_object_key == "jobs/result.json"