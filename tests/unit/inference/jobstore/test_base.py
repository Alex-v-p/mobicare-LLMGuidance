from __future__ import annotations

from collections import deque

import pytest

from inference.jobstore.base import JobStateConflictError, RedisJobStoreBase, managed_store
from shared.contracts.inference import InferenceRequest, JobRecord


class ClosableStore:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.queue: deque[str] = deque()
        self.blpop_timeouts: list[int] = []

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def rpush(self, queue_name, job_id):
        self.queue.append(job_id)
        return len(self.queue)

    async def get(self, key):
        value = self.values.get(key)
        return value if isinstance(value, str) else None

    async def blpop(self, queue_name, timeout=0):
        self.blpop_timeouts.append(timeout)
        if not self.queue:
            return None
        return queue_name, self.queue.popleft()

    async def zadd(self, key, mapping):
        bucket = self.values.setdefault(key, {})
        if not isinstance(bucket, dict):
            bucket = {}
            self.values[key] = bucket
        bucket.update(mapping)
        return len(bucket)

    async def zrem(self, key, member):
        bucket = self.values.get(key)
        if not isinstance(bucket, dict):
            return 0
        removed = member in bucket
        bucket.pop(member, None)
        return 1 if removed else 0

    async def zrangebyscore(self, key, min='-inf', max='+inf', start=0, num=None):
        bucket = self.values.get(key) or {}
        if not isinstance(bucket, dict):
            return []

        def _resolve(value):
            if value == '-inf':
                return float('-inf')
            if value == '+inf':
                return float('inf')
            return float(value)

        lower = _resolve(min)
        upper = _resolve(max)
        members = [
            member
            for member, score in sorted(bucket.items(), key=lambda item: (item[1], item[0]))
            if lower <= float(score) <= upper
        ]
        if num is None:
            return members[start:]
        return members[start : start + num]

    async def aclose(self):
        return None


class FakeJobStore(RedisJobStoreBase[JobRecord]):
    def __init__(self) -> None:
        super().__init__(
            model_cls=JobRecord,
            redis_url="redis://unused",
            queue_name="jobs",
            key_prefix="job:",
            ttl_seconds=3600,
            lease_seconds=30,
        )
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


@pytest.mark.asyncio
async def test_requeue_stale_running_jobs_moves_expired_lease_back_to_queue():
    store = FakeJobStore()
    record = JobRecord(request_id="req-2", status="queued", request=InferenceRequest(request_id="req-2", question="What now?"))
    await store.create(record)

    claimed = await store.claim_next(worker_id="worker-1", timeout_s=0)
    assert claimed is not None

    claimed.lease_expires_at = "2000-01-01T00:00:00+00:00"
    await store.update(claimed)

    requeued = await store.requeue_stale_running_jobs()
    loaded = await store.get(claimed.job_id)

    assert requeued == 1
    assert loaded is not None
    assert loaded.status == "queued"
    assert loaded.worker_id is None
    assert store.fake_client.queue[-1] == claimed.job_id


@pytest.mark.asyncio
async def test_mark_completed_rejects_worker_ownership_conflict():
    store = FakeJobStore()
    record = JobRecord(request_id="req-3", status="queued", request=InferenceRequest(request_id="req-3", question="What now?"))
    await store.create(record)

    claimed = await store.claim_next(worker_id="worker-1", timeout_s=0)
    assert claimed is not None

    loaded = await store.get(claimed.job_id)
    assert loaded is not None
    loaded.worker_id = "worker-2"
    await store.update(loaded)

    with pytest.raises(JobStateConflictError):
        await store.mark_completed(
            claimed,
            result=None,
            completed_at="2026-03-26T20:00:00+00:00",
            result_object_key="jobs/result.json",
        )


@pytest.mark.asyncio
async def test_claim_next_uses_nonzero_blpop_timeout_for_positive_wait(monkeypatch):
    store = FakeJobStore()
    monotonic_values = iter([100.0, 100.01])
    monkeypatch.setattr("inference.jobstore.base.monotonic", lambda: next(monotonic_values))

    claimed = await store.claim_next(worker_id="worker-1", timeout_s=1)

    assert claimed is None
    assert store.fake_client.blpop_timeouts == [1]
