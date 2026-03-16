from __future__ import annotations

from collections import deque

import pytest

from inference.jobstore.base import RedisJobStoreBase, managed_store
from shared.contracts.inference import InferenceRequest, JobRecord


class ClosableStore:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.queue: deque[str] = deque()

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def rpush(self, queue_name, job_id):
        self.queue.append(job_id)
        return len(self.queue)

    async def get(self, key):
        return self.values.get(key)

    async def blpop(self, queue_name, timeout=0):
        if not self.queue:
            return None
        return queue_name, self.queue.popleft()

    async def scan_iter(self, match=None):
        for key in list(self.values.keys()):
            yield key

    async def aclose(self):
        return None


class FakeJobStore(RedisJobStoreBase[JobRecord]):
    def __init__(self) -> None:
        super().__init__(model_cls=JobRecord, redis_url="redis://unused", queue_name="jobs", key_prefix="job:", key_pattern="job:*", ttl_seconds=3600, lease_seconds=30)
        self.fake_client = FakeRedis()

    async def _client(self):
        return self.fake_client


@pytest.mark.asyncio
async def test_managed_store_closes_store_after_use():
    store = ClosableStore()

    async with managed_store(lambda: store):
        pass

    assert store.closed is True


def test_validate_transition_rejects_invalid_state_change():
    store = FakeJobStore()

    with pytest.raises(ValueError):
        store._validate_transition("completed", "running")


@pytest.mark.asyncio
async def test_create_and_get_round_trip():
    store = FakeJobStore()
    record = JobRecord(request_id="req-1", status="queued", request=InferenceRequest(request_id="req-1", question="What now?"))

    await store.create(record)
    loaded = await store.get(record.job_id)

    assert loaded is not None
    assert loaded.job_id == record.job_id
    assert store.fake_client.queue[0] == record.job_id


@pytest.mark.asyncio
async def test_claim_next_sets_running_worker_and_lease():
    store = FakeJobStore()
    record = JobRecord(request_id="req-1", status="queued", request=InferenceRequest(request_id="req-1", question="What now?"))
    await store.create(record)

    claimed = await store.claim_next(worker_id="worker-1", timeout_s=0)

    assert claimed is not None
    assert claimed.status == "running"
    assert claimed.worker_id == "worker-1"
    assert claimed.lease_expires_at is not None


@pytest.mark.asyncio
async def test_heartbeat_returns_false_for_wrong_worker():
    store = FakeJobStore()
    record = JobRecord(request_id="req-1", status="queued", request=InferenceRequest(request_id="req-1", question="What now?"))
    await store.create(record)
    claimed = await store.claim_next(worker_id="worker-1", timeout_s=0)
    assert claimed is not None

    keep_going = await store.heartbeat(claimed.job_id, "worker-2")

    assert keep_going is False
